"""Stop only workers listed by one experiment launch record."""
import argparse
import json
import os
import signal
from pathlib import Path
from .launch import remote_stop


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run.resolve();records=json.loads((run/'launch.json').read_text())
    for node,row in records.items():
        if node!='A': continue
        pid=row['launcher_pid'];cmd=Path('/proc')/str(pid)/'cmdline'
        if cmd.exists() and str(run).encode() in cmd.read_bytes(): os.killpg(pid,signal.SIGTERM)
    remote_stop(run.name)
    print('Stop requested for',run.name)


if __name__=='__main__':main()
