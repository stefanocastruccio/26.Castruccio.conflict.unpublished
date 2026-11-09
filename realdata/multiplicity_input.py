# Builds the input of realdata/count_multiplicity.py from the raw partitions ($CLEAN has no per-type weekly counts). Row data stays in $WORK.
# Temporary input for realdata/count_multiplicity.py: weekly event counts by type from Luca's partitions
# (first copy per grid_id x week_start), ISO_A3 from $CLEAN. Row data stays in $WORK.
import glob, os, pandas as pd, pyarrow.csv as pc, pyarrow as pa
P = "/groups/scastruc/lmenical/drew/data/partitions"; W = os.environ["WORK"]
cols = ["week_start", "grid_id", "Battles", "ExplosionsRemote_violence", "Violence_against_civilians", "Protests", "Riots", "lon", "lat"]
parts = []
for f in sorted(glob.glob(f"{P}/data_final_world_*.csv")):
    t = pc.read_csv(f, convert_options=pc.ConvertOptions(include_columns=cols,
                    column_types={"week_start": pa.string(), "grid_id": pa.float64()})).to_pandas()
    n0 = len(t); t = t.drop_duplicates(["grid_id", "week_start"], keep="first")
    parts.append(t); print(os.path.basename(f), n0, "->", len(t), flush=True)
d = pd.concat(parts, ignore_index=True)
iso = pd.read_parquet(os.environ["CLEAN"], columns=["grid_id", "ISO_A3"]).drop_duplicates("grid_id")
d = d.merge(iso, on="grid_id", how="inner")
d["week_start"] = pd.to_datetime(d.week_start)
out = f"{W}/multiplicity_input.parquet"; d.to_parquet(out, index=False)
print("rows", len(d), "->", out)
