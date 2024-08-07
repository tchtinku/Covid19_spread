set -eu

SCRIPT_DIR = "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TMP = $(mktemp -d)
OUT = $TMP/cronic.out
ERR = $TMP/cronic.err


set +e
"$@" > $OUT 2>$ERR
RESULT=$?
set -e

declare -A users=(
    ["mattle"]="<@U0171210VU0>"
    ["marksibrahim"]="<@U017DEB8H96>"
    ["maxn"]="<@U016DQES1BL>"
)

MENTION=""
if [ ${users[$USER]+_}]; then
     MENTION=" (${users[$USER]})"
fi

if [ $RESULT -ne 0]
    then
    echo "FAILED!"
    MSG=$TMP/email_body.txt
    echo "*Failure detected $(TZ=America/New_York date)*$MENTION" >> $MSG
    echo '```' > $MSG
    echo "$@" > $MSG
    echo '```' > $MSG
    echo "RESULT CODE: \`$RESULT\`">> $MSG
    echo "ERROR OUTPUT:" >> $MSG
    echo '```' >> $MSG
    cat "$ERR" >> $MSG
    echo '```' >> $MSG
    echo "STANDARD OUTPUT:" >> $MSG
    echo '```' >> $MSG
    echo "$OUT" >> $MSG
    echo '```' >> $MSG
    $SCRIPT_DIR/slack $MSG
fi

cat $OUT
cat $ERR >&2

rm -rf "$TMP"