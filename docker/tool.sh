#!/usr/bin/bash
 
COMMAND=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
DOCKER_TAG="$2"
 
DOCKER_BINARY="docker"
DOCKERFILE="docker/Dockerfile"
 
# print arguments
echo "DOCKER_BINARY: ${DOCKER_BINARY}"
echo "DOCKERFILE: ${DOCKERFILE}"
echo "DOCKER_TAG: ${DOCKER_TAG}"
 
# using parent folder to build mirror
cd ..
 
# build mirror
if [[ "${COMMAND}" == "build" ]]; then
    ${DOCKER_BINARY} build -t ${DOCKER_TAG} -f ${DOCKERFILE} .
elif [[ "${COMMAND}" == "push" ]]; then
    ${DOCKER_BINARY} push ${DOCKER_TAG}
else
    echo "Unknow COMMAND=${COMMAND}"
    exit 1
fi