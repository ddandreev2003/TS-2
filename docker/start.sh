#!/bin/bash
set -euo pipefail
/docker/entrypoint.sh supervisord -c /docker/supervisord.conf
