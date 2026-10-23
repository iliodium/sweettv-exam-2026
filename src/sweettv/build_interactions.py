"""Construct temporally separated programme-airing coverage from immutable logs."""
from pathlib import Path
import argparse,json,time,duckdb
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=ap.parse_args()
R=args.root;t=time.perf_counter();c=duckdb.connect(str(R/'data/processed/data.duckdb'))
c.execute("SET memory_limit='10GB'");c.execute('SET threads=8')
train_start='2020-03-09 00:00:00';cutoff='2020-06-29 00:00:00';test_start='2020-07-27 00:00:00';test_end='2020-10-19 00:00:00'
c.execute('CREATE OR REPLACE TABLE windows(split VARCHAR, start_at TIMESTAMP, end_at TIMESTAMP)')
c.executemany('INSERT INTO windows VALUES (?,?,?)',[('train',train_start,cutoff),('val',cutoff,test_start)])
c.execute('''CREATE OR REPLACE TABLE window_airings AS SELECT w.split,a.* FROM airings a JOIN windows w
 ON a.start_at>=w.start_at AND a.end_at<=w.end_at WHERE a.source_part<2''')
print('Starting channel/interval join on labelled airings only',flush=True)
c.execute('''CREATE OR REPLACE TABLE joined_segments AS SELECT a.split,v.user_id,v.view_id,a.airing_id,a.item_id,
 a.channel_id,a.duration_seconds,a.start_at AS airing_start,a.end_at AS airing_end,
 greatest(v.start_at,a.start_at) AS overlap_start,least(v.stop_at,a.end_at) AS overlap_end
 FROM window_airings a JOIN view_log v ON v.channel_id=a.channel_id
 AND v.start_at<a.end_at AND v.stop_at>a.start_at''')
print('Joined segments: '+str(c.execute('SELECT count(*) FROM joined_segments').fetchone()[0]),flush=True)
c.execute('''CREATE OR REPLACE TABLE airing_watch AS WITH previous AS (
 SELECT *,max(overlap_end) OVER(PARTITION BY split,user_id,airing_id ORDER BY overlap_start,overlap_end,view_id
 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM joined_segments),
 grouped AS (SELECT *,sum(CASE WHEN previous_end IS NULL OR overlap_start>previous_end THEN 1 ELSE 0 END)
 OVER(PARTITION BY split,user_id,airing_id ORDER BY overlap_start,overlap_end,view_id ROWS UNBOUNDED PRECEDING) AS grp FROM previous),
 islands AS (SELECT split,user_id,airing_id,item_id,channel_id,duration_seconds,airing_start,airing_end,grp,
 min(overlap_start) AS start_at,max(overlap_end) AS end_at FROM grouped
 GROUP BY split,user_id,airing_id,item_id,channel_id,duration_seconds,airing_start,airing_end,grp)
 SELECT split,user_id,airing_id,item_id,channel_id,duration_seconds,airing_start,airing_end,
 sum(date_diff('second',start_at,end_at)) AS watch_seconds,
 sum(date_diff('second',start_at,end_at))::DOUBLE/duration_seconds AS coverage
 FROM islands GROUP BY split,user_id,airing_id,item_id,channel_id,duration_seconds,airing_start,airing_end''')
assert c.execute('SELECT count(*) FROM airing_watch WHERE coverage<0 OR coverage>1.0000001').fetchone()[0]==0
c.execute("COPY airing_watch TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(R/'data/processed/airing_watch.parquet')])
c.execute("COPY (SELECT * FROM airing_watch WHERE coverage>=0.8) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(R/'data/processed/events.parquet')])
# Metadata fitting scope is controlled by each model: only X-supported training item rows.
# Earliest known metadata is retained; later schedule is used for inference/availability only.
c.execute('''CREATE OR REPLACE TABLE item_metadata AS SELECT item_id,
 first(title ORDER BY source_part,start_at) AS title,first(category ORDER BY source_part,start_at) AS category,
 first(genre_1 ORDER BY source_part,start_at) AS genre_1,first(genre_2 ORDER BY source_part,start_at) AS genre_2,
 first(genre_3 ORDER BY source_part,start_at) AS genre_3,first(year ORDER BY source_part,start_at) AS year,
 first(director ORDER BY source_part,start_at) AS director,first(actors ORDER BY source_part,start_at) AS actors
 FROM schedule_all WHERE item_id>0 GROUP BY item_id ORDER BY item_id''')
metadata=c.execute('SELECT * FROM item_metadata ORDER BY item_id').fetchdf().fillna('').to_dict(orient='records')
(R/'data/processed/metadata.json').write_text(json.dumps(metadata,ensure_ascii=False))
# A programme is eligible only if its labelled airing is available in the target period.
val_ids=[x[0] for x in c.execute('SELECT DISTINCT item_id FROM airings WHERE source_part<2 AND start_at>=? AND start_at<? ORDER BY item_id',[cutoff,test_start]).fetchall()]
test_ids=[x[0] for x in c.execute('SELECT DISTINCT item_id FROM airings WHERE source_part=2 AND start_at>=? AND start_at<? ORDER BY item_id',[test_start,test_end]).fetchall()]
(R/'data/processed/candidates.json').write_text(json.dumps({'val':val_ids,'test':test_ids}))
# Unknown-id airings remain excluded as programmes; train-only channel seconds provide optional context.
c.execute(f"""COPY (SELECT user_id,channel_id,
 sum(date_diff('second',greatest(start_at,CAST(? AS TIMESTAMP)),least(stop_at,CAST(? AS TIMESTAMP)))) AS view_seconds
 FROM view_log WHERE start_at<CAST(? AS TIMESTAMP) AND stop_at>CAST(? AS TIMESTAMP)
 GROUP BY user_id,channel_id) TO '{R / 'data/processed/train_channel_context.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""",[train_start,cutoff,cutoff,train_start])
# Per-item/channel target schedule supply uses no user viewing information.
c.execute(f"""COPY (SELECT 'val' AS split,item_id,channel_id,count(*) AS airing_count,sum(duration_seconds) AS schedule_seconds
 FROM airings WHERE source_part<2 AND start_at>=CAST(? AS TIMESTAMP) AND start_at<CAST(? AS TIMESTAMP) GROUP BY item_id,channel_id UNION ALL
 SELECT 'test' AS split,item_id,channel_id,count(*) AS airing_count,sum(duration_seconds) AS schedule_seconds
 FROM airings WHERE source_part=2 AND start_at>=? AND start_at<? GROUP BY item_id,channel_id)
 TO '{R / 'data/processed/target_channel_supply.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""",[cutoff,test_start,test_start,test_end])
def query(q,params=None):
 cur=c.execute(q,params or []);names=[x[0] for x in cur.description];return [dict(zip(names,row)) for row in cur.fetchall()]
out={'protocol':{'train_start':train_start,'validation_start':cutoff,'validation_end':test_start,'hidden_test_start':test_start,'hidden_test_end':test_end,'intervals':'half-open; require each airing fully inside its fold; union overlapping view intervals per user/airing','candidate_availability':'airing START within target window, consistent VAL/TEST; source files0/1 for VAL and2 for TEST; unlike labels, candidate airing may end across boundary','positive_threshold':0.8,'repeats_allowed':True,'validation_target':'five most frequent qualifying distinct programmes per user; count distinct high-coverage airings','fitting_validation_views':False,'time_zone':'timestamps as supplied; no timezone offset invented'},
'coverage':query('''SELECT split,count(*) matched_user_airings,count(DISTINCT user_id) matched_users,
 count(*) FILTER(WHERE coverage>=.8) positive_user_airings,count(DISTINCT CASE WHEN coverage>=.8 THEN user_id END) positive_users,
 count(DISTINCT CASE WHEN coverage>=.8 THEN item_id END) positive_items,
 quantile_cont(coverage,[.1,.5,.8,.9]) coverage_quantiles FROM airing_watch GROUP BY split ORDER BY split'''),
'candidate_counts':{'val':len(val_ids),'test':len(test_ids),'metadata_universe':len(metadata)},
'raw_joins':query('SELECT split,count(*) overlap_segments,count(DISTINCT user_id) users,count(DISTINCT airing_id) airings FROM joined_segments GROUP BY split ORDER BY split'),
'boundary_exclusions':query('''SELECT w.split,count(*) boundary_airings FROM windows w JOIN airings a ON a.source_part<2
 AND a.start_at<w.end_at AND a.end_at>w.start_at WHERE a.start_at<w.start_at OR a.end_at>w.end_at GROUP BY w.split'''),
'normalization_checks':query('SELECT min(coverage) min_coverage,max(coverage) max_coverage,count(*) FILTER(WHERE watch_seconds>duration_seconds) impossible_coverage FROM airing_watch'),
'duration_mismatch':query('SELECT min(declared_seconds-date_diff(\'second\',start_at,stop_at)) min_delta,max(declared_seconds-date_diff(\'second\',start_at,stop_at)) max_delta FROM view_log WHERE declared_seconds<>date_diff(\'second\',start_at,stop_at)'),
'seconds':time.perf_counter()-t}
(R/'evidence/interaction_profile.json').write_text(json.dumps(out,default=str,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,default=str,ensure_ascii=False,indent=2));c.close()
