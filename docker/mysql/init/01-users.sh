#!/bin/bash
# Runs once, on first initialisation of the data volume.
# Creates the main + test databases and two least-privilege users:
#   app      -> DML only
#   migrator -> DDL + DML, scoped to the two databases (no global privileges)
set -euo pipefail

mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE DATABASE IF NOT EXISTS \`${MYSQL_DATABASE}\`
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS \`${MYSQL_TEST_DATABASE}\`
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

CREATE USER IF NOT EXISTS '${MYSQL_APP_USER}'@'%' IDENTIFIED BY '${MYSQL_APP_PASSWORD}';
GRANT SELECT, INSERT, UPDATE, DELETE ON \`${MYSQL_DATABASE}\`.*      TO '${MYSQL_APP_USER}'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE ON \`${MYSQL_TEST_DATABASE}\`.* TO '${MYSQL_APP_USER}'@'%';

CREATE USER IF NOT EXISTS '${MYSQL_MIGRATOR_USER}'@'%' IDENTIFIED BY '${MYSQL_MIGRATOR_PASSWORD}';
GRANT ALL PRIVILEGES ON \`${MYSQL_DATABASE}\`.*      TO '${MYSQL_MIGRATOR_USER}'@'%';
GRANT ALL PRIVILEGES ON \`${MYSQL_TEST_DATABASE}\`.* TO '${MYSQL_MIGRATOR_USER}'@'%';

FLUSH PRIVILEGES;
SQL
