from pathlib import Path
from zipfile import ZipFile
import argparse,json,hashlib,time,csv,duckdb
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--archive',type=Path,required=True);args=ap.parse_args()
R=args.root;raw=R/'data/raw';db=R/'data/processed/data.duckdb';t=time.perf_counter();archive=args.archive
for folder in [raw,R/'data/processed',R/'evidence',R/'artifacts',R/'submission']:
 folder.mkdir(parents=True,exist_ok=True)
expected={'dataset11-30.csv','export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv','submission.csv'}
manifest={}
with ZipFile(archive) as z:
 assert set(z.namelist())==expected,'Unexpected archive contents'
 for name in sorted(expected):
  dest=raw/name
  if not dest.exists():
   with z.open(name) as f,dest.open('wb') as o:
    while chunk:=f.read(1024*1024):o.write(chunk)
  manifest[name]={'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
(R/'evidence/data_manifest.json').write_text(json.dumps({'source':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':manifest},indent=2))
c=duckdb.connect(str(db));c.execute("SET memory_limit='8GB'");c.execute('SET threads=8')
if 'view_log' not in {x[0] for x in c.execute('SHOW TABLES').fetchall()}:
 c.execute("""CREATE TABLE view_log AS SELECT row_number() OVER () AS view_id,
 user_id, CAST(vsetv_id AS INTEGER) channel_id,
 strptime(start_time,'%Y-%m-%d %H:%M:%S') AS start_at,
 strptime(stop_time,'%Y-%m-%d %H:%M:%S') AS stop_at,
 CAST(duraton AS BIGINT) declared_seconds FROM read_csv(?,header=true,all_varchar=true)""",[str(raw/'dataset11-30.csv')])
 rows=[]
 for idx,name in enumerate(['export_arh_11-20-final.csv','export_arh_21-30-final.csv','export_arh_31-42-final.csv']):
  c.execute(f"""CREATE TABLE schedule_{idx} AS SELECT CAST(channel_id AS INTEGER) channel_id,channel_title,
  strptime(start_time,'%d.%m.%Y %H:%M:%S') AS start_at,CAST(duration AS BIGINT) duration_seconds,
  tv_show_title AS title,CAST(tv_show_id AS INTEGER) AS item_id,tv_show_category AS category,
  tv_show_genre_1 AS genre_1,tv_show_genre_2 AS genre_2,tv_show_genre_3 AS genre_3,
  year_of_production AS year,director,actors,{idx} AS source_part
  FROM read_csv(?,header=true,all_varchar=true)""",[str(raw/name)])
 c.execute('CREATE VIEW schedule_all AS SELECT * FROM schedule_0 UNION ALL SELECT * FROM schedule_1 UNION ALL SELECT * FROM schedule_2')
 c.execute("""CREATE TABLE airings AS SELECT row_number() OVER() AS airing_id,* FROM (
 SELECT DISTINCT channel_id,start_at,start_at+duration_seconds*INTERVAL '1 SECOND' AS end_at,duration_seconds,item_id,source_part
 FROM schedule_all WHERE item_id>0 AND duration_seconds>0)""")
queries={
'view_log':"""SELECT count(*) AS n_rows,count(DISTINCT user_id) users,count(DISTINCT channel_id) channels,
 min(start_at) start_min,max(stop_at) stop_max,min(declared_seconds) duration_min,
 quantile_cont(declared_seconds,[.1,.5,.9,.99]) duration_quantiles,max(declared_seconds) duration_max,
 count(*) FILTER (WHERE start_at>=stop_at) invalid_intervals,
 count(*) FILTER(WHERE declared_seconds<>date_diff('second',start_at,stop_at)) declared_duration_mismatch
 FROM view_log""",
'schedules':"""SELECT source_part,count(*) AS n_rows,count(DISTINCT channel_id) channels,
 count(DISTINCT item_id) FILTER(WHERE item_id>0) programmes,
 count(*) FILTER(WHERE item_id>0) labelled_rows,count(*) FILTER(WHERE item_id=0) unlabelled_rows,
 min(start_at) start_min,max(start_at+duration_seconds*INTERVAL '1 SECOND') end_max,
 quantile_cont(duration_seconds,[.1,.5,.9,.99]) duration_quantiles,
 count(*) FILTER(WHERE duration_seconds<=0) invalid_duration
 FROM schedule_all GROUP BY source_part ORDER BY source_part""",
'airings':"SELECT source_part,count(*) distinct_labelled_airings,min(start_at) start_min,max(end_at) end_max FROM airings GROUP BY source_part ORDER BY source_part",
'weekly_views':"SELECT strftime(start_at,'%G-%V') AS week_id,count(*) AS n_rows,count(DISTINCT user_id) users FROM view_log GROUP BY week_id ORDER BY week_id",
'channel_mapping':"SELECT count(DISTINCT v.channel_id) channels_in_views,count(DISTINCT CASE WHEN s.channel_id IS NOT NULL THEN v.channel_id END) channels_with_schedule FROM (SELECT DISTINCT channel_id FROM view_log) v LEFT JOIN (SELECT DISTINCT channel_id FROM schedule_all) s USING(channel_id)",
'ambiguous_start':"SELECT count(*) conflicting_channel_start FROM (SELECT channel_id,start_at,count(DISTINCT item_id) items FROM airings GROUP BY channel_id,start_at HAVING items>1)"
}
out={}
for key,q in queries.items():
 res=c.execute(q);names=[d[0] for d in res.description];out[key]=[dict(zip(names,row)) for row in res.fetchall()]
with (raw/'submission.csv').open() as f:
 reader=csv.DictReader(f);users=[x['user_id'] for x in reader]
 assert len(users)==len(set(users));out['template']={'rows':len(users),'columns':reader.fieldnames,'user_ids_preserved_as_strings':True}
out['seconds']=time.perf_counter()-t;out['duckdb_version']=duckdb.__version__
(R/'evidence/raw_profile.json').write_text(json.dumps(out,default=str,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,default=str,ensure_ascii=False,indent=2));c.close()
