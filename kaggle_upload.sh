#!/bin/bash
# Auto-upload myna-repo to Kaggle as a code archive, then submit notebook job
# Usage: ./kaggle_upload.sh [--config scaled] [--steps 8000] [--batch 16]

set -e

CONFIG="${1:-v0}"
STEPS="${2:-8000}"
BATCH="${3:-16}"
STEP_NUM="$4"

cd /Users/aashish/next\ ko\ ne\ next\ gen\ model

echo "Packaging repo for Kaggle..."
rm -f myna_repo.tar.gz
tar -czf myna_repo.tar.gz . --exclude='.venv' --exclude='.git' --exclude='*.tar.gz' --exclude='runs/*' --exclude='data/**' --exclude='kaggle-dataset/**'

echo "Creating Kaggle dataset (code archive)..."
DATASET_NAME="aashish254/myna-7b-plus-$[STEP_NUM+1]"
if kaggle datasets version -p myna-7b-plus-$[STEP_NUM+1] --title "myna v$[(STEP_NUM+1)/1000]." 2>&1 | grep -q "already exists"; then
    echo "Dataset $DATASET_NAME already exists; deleting and recreating..."
    kaggle datasets delete -d $DATASET_NAME
fi
kaggle datasets create -r tar myna_repo.tar.gz --title "myna v$[(STEP_NUM+1)/1000.] config=$CONFIG steps=$STEPS batch=$BATCH"

echo "Submitting Kaggle notebook job..."
NOTEBOOK="notebooks/v$[(STEP_NUM+1)/1000].ipynb"
SUBMISSION=$(kaggle notebooks submit -f "$NOTEBOOK" --no-progress-bar --title "myna $[(STEP_NUM+1)]-scaled" --dataset "$DATASET_NAME:code" 2>&1 | grep -o '[a-f0-9]\{32\}' | head -1)
if [ -z "$SUBMISSION" ]; then
    echo "Failed to extract submission ID from:"
    kaggle notebooks submit -f "$NOTEBOOK" --no-progress-bar --title "myna $[(STEP_NUM+1)]-scaled" --dataset "$DATASET_NAME:code"
    exit 1
fi

echo "Submission ID: $SUBMISSION"
echo "Monitoring progress..."
while true; do
    STATUS=$(kaggle notebooks status $SUBMISSION | grep -oE 'COMPLETED|RUNNING|ERROR')
    echo "Status: $STATUS"
    if [[ "$STATUS" == "COMPLETED" ]]; then
        echo "Training completed! Downloading artifacts..."
        kaggle datasets download -p myna-7b-plus-$[STEP_NUM+1].tar.gz -d $DATASET_NAME --force
        tar -xzf myna-7b-plus-$[STEP_NUM+1].tar.gz
        mv runs/v$[(STEP_NUM+1)]_scaled_* /private/tmp/kgwork/ckptfull/runs/v$[(STEP_NUM+1)]_scaled/
        echo "Artifacts saved to /private/tmp/kgwork/ckptfull/runs/v$[(STEP_NUM+1)]_scaled/"
        break
    elif [[ "$STATUS" == "ERROR" ]]; then
        echo "Training failed!"
        kaggle notebooks status $SUBMISSION
        exit 1
    fi
    sleep 300
done
