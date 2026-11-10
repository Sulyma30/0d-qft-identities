#!/bin/bash
# Runs all checks and writes the outputs to logs/. Total run time about 3 minutes.
mkdir -p logs
for s in single_field two_fields two_fields_parquet two_fields_five_point fermions fermions_three_flavours sbe_residual; do
    echo "running $s.py"
    python3 $s.py > logs/$s.log 2>&1
    tail -1 logs/$s.log
done

echo "running appendixC.py 4"
python3 appendixC.py 4 > logs/appendixC_D4.log 2>&1
tail -1 logs/appendixC_D4.log

echo "FAIL lines in logs:"; grep -l "^FAIL" logs/*.log || echo "none"
