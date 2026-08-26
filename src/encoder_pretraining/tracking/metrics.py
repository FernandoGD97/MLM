"""Append-only tidy epoch metrics with JSON summary."""
from pathlib import Path
import json

class MetricsTracker:
    COLUMNS=("epoch","global_step","processed_tokens","objective","metric_group","dataset","metric","value")
    def __init__(self,run_dir):
        self.run_dir=Path(run_dir); self.rows=[]
        existing=self.run_dir/"epoch_metrics.parquet"
        if existing.exists():
            import pyarrow.parquet as pq
            self.rows=pq.read_table(existing).to_pylist()
    def add_metrics(self,epoch,global_step,processed_tokens,objective,group,dataset,metrics):
        for metric,value in metrics.items():
            if isinstance(value,(int,float)):
                self.rows.append(dict(epoch=epoch,global_step=global_step,processed_tokens=processed_tokens,objective=objective,
                                      metric_group=group,dataset=dataset,metric=metric,value=float(value)))
        self.flush()
    def flush(self):
        import pyarrow as pa, pyarrow.parquet as pq
        table=pa.Table.from_pylist(self.rows,schema=pa.schema([("epoch",pa.int64()),("global_step",pa.int64()),("processed_tokens",pa.int64()),
            ("objective",pa.string()),("metric_group",pa.string()),("dataset",pa.string()),("metric",pa.string()),("value",pa.float64())]))
        pq.write_table(table,self.run_dir/"epoch_metrics.parquet")
        latest={f'{x["metric_group"]}/{x["dataset"]}/{x["metric"]}':x["value"] for x in self.rows if x["epoch"]==max((r["epoch"] for r in self.rows),default=0)}
        (self.run_dir/"epoch_summary.json").write_text(json.dumps(latest,indent=2,sort_keys=True))
