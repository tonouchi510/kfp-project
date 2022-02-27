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
    -h|--help)
      echo "-p: pipeline name"
      echo "-b: bucket name"
      echo "-j: job id (dataset file name)"
      echo "--msg: message string to notify slack."
      exit
      ;;
    -p|--pipeline_name)
      export PIPELINE_NAME=$2
      shift 2
      ;;
    -b|--bucket_name)
      export BUCKET=$2
      shift 2
      ;;
    -j|--job_id)
      export JOB_ID=$2
      shift 2
      ;;
    --message)
      export MESSAGE=$2
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

# 将来的にSecret Managerへのアクセス権限与えるので、そこからとるようにする.
#URL=$(gcloud secrets versions access latest --secret="kfp-slack-webhook-url")

# WebHookのURL
URL=$(gcloud secrets versions access latest --secret="kfp-slack-webhook-url" --quiet)
# 送信先のチャンネル
CHANNEL=${CHANNEL:-'#dev-notify'}
# botの名前
BOTNAME=${BOTNAME:-'kfp-bot'}
# 絵文字
EMOJI=${EMOJI:-':moyai:'}
# 見出し
HEAD=${HEAD:-"[${PIPELINE_NAME}: ${JOB_ID}]\n"}

# メッセージをシンタックスハイライト付きで取得
MESSAGE='```'${MESSAGE}'```'

# json形式に整形
payload="payload={
    \"channel\": \"${CHANNEL}\",
    \"username\": \"${BOTNAME}\",
    \"icon_emoji\": \"${EMOJI}\",
    \"text\": \"${HEAD}${MESSAGE}\"
}"

# 送信
curl -s -S -X POST --data-urlencode "${payload}" --insecure ${URL} > /dev/null
