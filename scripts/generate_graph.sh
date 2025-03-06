#!/bin/bash

usage() {
    echo "Usage: $0 -i INPUT -d DURATION -u INTERVAL"
    exit 1
}

# 解析命令行参数
while getopts "i:d:u:" opt; do
    case $opt in
        i)
            INPUT=$OPTARG
            ;;
        d)
            DURATION=$OPTARG
            ;;
        u)
            INTERVAL=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# 检查必要的参数是否已提供
if [ -z "$INPUT" ] || [ -z "$DURATION" ] || [ -z "$INTERVAL" ]; then
    usage
fi


HOME_DIR="/path/to/MPI_profile"
INPUT_DIR="${HOME_DIR}/${INPUT}"
OUTPUT_DIR="${HOME_DIR}/${INPUT}/${DURATION}ms_closed"

echo "Running analysis with the following parameters:"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Duration: ${DURATION}ms"
echo "Interval: ${INTERVAL}ms"

echo "Duration: $(( DURATION * 2500000 ))"
echo "Interval: $(( INTERVAL * 2500000 ))"

mkdir ${OUTPUT_DIR}


timeslice_analysis -i ${INPUT_DIR}/trace -f -o ${OUTPUT_DIR}/graph -d ${INPUT_DIR}/lineinfo -l $((INTERVAL * 2500000 )) -u $(( DURATION * 2500000 ))
timeslice_analysis -i ${INPUT_DIR}/trace -f -o ${OUTPUT_DIR}/graph_edge -d ${INPUT_DIR}/lineinfo -l $(( INTERVAL * 2500000 )) -u $(( DURATION * 2500000 )) -e
