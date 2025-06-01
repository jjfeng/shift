"""
a script for wrapping around my python scripts
"""
import os
import time
import sys
from pathlib import Path
import argparse
import logging
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
        description="run code on the cluster or locally"
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=300
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default='local',
        choices=['local', 'cluster']
    )
    parser.add_argument(
        "--target-template-files",
        type=str,
    )
    parser.add_argument(
        "--is-short",
        action="store_true",
    )
    parser.add_argument(
        "--run-line",
        type=str,
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
    )
    args = parser.parse_args()
    args.target_template_files = args.target_template_files.split(",")
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    # figure out missing jobs
    missing_jobs = []
    for job_idx in range(1, args.num_jobs + 1):
        job_completed = True
        for target_template_file in args.target_template_files:
            target_file = target_template_file.replace("JOB", str(job_idx))
            print("checking for target file", target_file)
            if not os.path.exists(target_file):
                job_completed = False
                break
        if not job_completed:
            missing_jobs.append(job_idx)
    print("missing jobs", missing_jobs)

    if args.cluster == 'local':
        for i in missing_jobs:
            run_cmd = "python %s --job %d" % (args.run_line, i)
            print(run_cmd)
            subprocess.check_output(
                run_cmd, stderr=subprocess.STDOUT, shell=True
            )
    else:
        run_script = "run_script.sh" if args.is_short else "run_script_long.sh"
        if len(missing_jobs) == 0:
            print("dont need to run qsub")
        elif (max(missing_jobs) - min(missing_jobs)) != (len(missing_jobs) - 1):
            # if the missing jobs does not correspond to a straightforward sequence
            for job_idx in missing_jobs:
                qsub_cmd = ("qsub -t %d-%d %s %s" % (job_idx, job_idx, run_script, args.run_line))
                print(qsub_cmd)
                output = subprocess.check_output(
                    qsub_cmd,
                    stderr=subprocess.STDOUT,
                    shell=True,
                )
                print("QSUB DONE", output)
        else:
            qsub_cmd = ("qsub -t %d-%d -tc %d %s %s" % (min(missing_jobs),
                max(missing_jobs), args.num_parallel, run_script, args.run_line))
            print(qsub_cmd)
            output = subprocess.check_output(
                qsub_cmd,
                stderr=subprocess.STDOUT,
                shell=True,
            )
            print("QSUB DONE", output)

    # Check that the desired files are in the file system.
    wait_iters = 0
    for t in range(20 * args.num_jobs):
        jobs_completed = []
        for job_idx in missing_jobs:
            job_completed = True
            for target_template_file in args.target_template_files:
                target_file = target_template_file.replace("JOB", str(job_idx))
                print("checking for target file", target_file)
                if not os.path.exists(target_file):
                    job_completed = False
                    break
            if job_completed:
                jobs_completed.append(job_idx)

        do_finish = (len(jobs_completed) == len(missing_jobs))
        if len(jobs_completed) > 0.9 * len(missing_jobs):
            wait_iters += 1
            if wait_iters > 6:
                do_finish = True
        if do_finish:
            # If I have been waiting and the number of results hasn't
            # increased, just consider the job completed and move onwards.
            # Create files to indicate to scons that the job is done
            for target_template_file in args.target_template_files:
                Path(target_template_file).touch()
            break
        else:
            time.sleep(30)

    time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:])
