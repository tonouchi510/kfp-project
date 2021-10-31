#!/bin/bash
set -e

echo "                                                                          "
echo "              =======================================                     "
echo "              ==    Slack Notification Component   ==                     "
echo "              =======================================                     "
echo "                                                                          "

for OPT in "$@"
do
  case "$OPT" in
    --pipeline_name)
      export PIPELINE_NAME=$2
      shift 2
      ;;
    --job_id)
      export JOB_ID=$2
      shift 2
      ;;
    -\?)
      exit 1
      ;;
    -*)
      echo "$0: illegal option -- $1" >&2
      exit 1
      ;;
  esac
done

# WebHookのURL
URL=$(gcloud secrets versions access latest --project=furyu-nbiz --secret="kfp-slack-webhook-url" --quiet)
# 送信先のチャンネル
CHANNEL=${CHANNEL:-'#dev-notify'}
# botの名前
BOTNAME=${BOTNAME:-'kfp-bot'}
# 絵文字
EMOJI=${EMOJI:-':moyai:'}
# メッセージをシンタックスハイライト付きで取得
MESSAGE="Done ${PIPELINE_NAME}: ${JOB_ID}"

# json形式に整形
payload="payload={
    \"channel\": \"${CHANNEL}\",
    \"username\": \"${BOTNAME}\",
    \"icon_emoji\": \"${EMOJI}\",
    \"text\": \"${MESSAGE}\"
}"

# 送信
curl -s -S -X POST --data-urlencode "${payload}" --insecure ${URL} > /dev/null

echo "Done."
