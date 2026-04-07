#!/bin/bash
set -e

echo "=== Spider Text-to-SQL Evaluation ==="
echo "MODAL_PLANNER_URL: $MODAL_PLANNER_URL"

python test/text2sql/agentflow_spider.py \
    --engine modal \
    --db_dir data/spider/database
