for var in "$@"
do
    sbatch --job-name="SimPle-job-$var" simple_job.sh
done