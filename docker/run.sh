#!/usr/bin/bash
 
function show_usage() {
    echo ""
    echo "Usage: bash run.sh DOCKER_TAG ENV"
    echo "   * DOCKER_TAG: docker tag, e.g. v1.0.0"
    echo "   * ENV: Optional(default: TEST)"
    echo "   **  TEST: kd-bd02.kuandeng.com/kd-recog/recog"
    echo "   **  PROD: op-01.gzproduction.com/kd-recog/recog"
    echo ""
}
 
if (( $# < 1 )); then
    show_usage
    exit -1
fi
 
DOCKER_TAG="$1"
 
SERVER="kd-bd02.kuandeng.com/kd-recog/platform"
if (( $# == 2 )); then
    if [ "$2" = "TEST" ];
        then SERVER="kd-bd02.kuandeng.com/kd-recog/platform"
    elif [ "$2" = "PROD" ];
        then SERVER="op-01.gzproduction.com/kd-recog/platform"
    else
        echo "\n========== ERROR! Unknown environment given : $2 ============"
        show_usage
        exit 1
    fi
fi
 
DOCKER_NAME="${SERVER}:${DOCKER_TAG}"
echo "DOCKER NAME:${DOCKER_NAME}"
 
echo ""
echo "******* build image *********************************************************"
bash tool.sh build ${DOCKER_NAME}
 
echo ""
echo "******* push image **********************************************************"
bash tool.sh push ${DOCKER_NAME}
 
echo ""
echo "***** finish ******"