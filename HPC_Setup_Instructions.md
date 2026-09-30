# HPC Setup Instructions

> Note: Git is already installed system-wide on the HPC.

## 1. Clone the repository

Make sure you are in your HPC home directory. You can check your current directory with:

```bash
pwd
```

Then clone the repository:

```bash
git clone https://github.com/ScripterDude/Course-Project-2-Video-Classification
```

Move into the project directory:

```bash
cd Course-Project-2-Video-Classification
```

## 2. Create and activate a virtual environment

Create a Python virtual environment:

```bash
python3 -m venv venv
```

Activate it: (You probably need to activate this each time you log on)

```bash
source venv/bin/activate
```

## 3. Install the required libraries

Install the project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
Youre basicly set up now.

# Utility commands

## To update code on HPC pull command

```bash
git pull origin main
```

## Activate venv each time on logon.

```bash
source venv/bin/activate
```
# Submit and manage batch jobs

Model training should be submitted to the **course-specific batch queue/resources**.

You do not manually enter a batch node. The LSF scheduler assigns the job to the available course compute resources.

The course provides the `c02516` batch resources with GPU slices of 10 GB each.

### Submit a batch job

```bash
bsub -app c02516_1g.10gb < jobscript.sh
```

This submits `jobscript.sh` using one 10 GB GPU slice.

### Check your jobs

```bash
bstat
```

### Stop a job

```bash
bkill JOBID
```

Replace `JOBID` with the job ID shown by `bstat`.

### Quick reference

```bash
bsub -app c02516_1g.10gb < jobscript.sh  # Submit job
bstat                                    # Check job status
bkill JOBID                              # Stop job
```