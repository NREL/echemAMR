#!/bin/bash 

rm -r outputs
mkdir outputs

paddings="256 128 64 32 16 8 4 2 1"
# paddings="16 8"

first=True
for pad in $paddings; do
    ./main2d.llvm.TEST.MPI.ex m_pad=$pad n_pad=$pad plot_file=outputs/plots_$pad 2>&1 | tee outputs/run_$pad.txt
    if [ $first = True ]; then
        echo "Padding Size: $pad" > outputs/compiled.txt
        tail -n 4 outputs/run_$pad.txt | head -n 2 >> outputs/compiled.txt
        first=False
    else
        echo "Padding Size: $pad" >> outputs/compiled.txt
        tail -n 4 outputs/run_$pad.txt | head -n 2 >> outputs/compiled.txt
    fi
    echo "" >> outputs/compiled.txt
done

