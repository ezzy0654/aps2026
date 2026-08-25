#!/usr/bin/env bash
# APS 프로젝트 성능 자동 제출
#
#   ./submit.sh                    # ./run.sh -n 1 실행 후 결과 자동 제출
#   ./submit.sh -n 1 -d            # run.sh 에 그대로 전달되는 옵션
#   ./submit.sh --note "fused v3"  # 메모를 붙여 제출
#   ./submit.sh --dry-run          # 실행만 하고 제출은 하지 않음
#   ./submit.sh --from-log run.log # 이미 받아둔 실행 로그를 그대로 제출
#   ./submit.sh --no-update        # 자기 갱신 없이 이 사본 그대로 실행
#   ./submit.sh --as aps07         # [조교] aps07 이름으로 대신 등록 (관리자 키 필요)
#
# 프로젝트 폴더(run.sh 가 있는 곳)에 두고 실행하세요.
#
# 이 스크립트가 하는 일
#   1. run.sh 로 slurm 작업 제출 → 출력 전체를 로그로 저장
#   2. 로그에서  "Throughput: <값> (<단위>)"  를 읽음
#   3. 수강반 판정 (run.sh 옵션 기준)
#        -d 있음 (Decode: ON)  → 고급반 · tokens_per_s (= sequences/sec × 8토큰)
#        -d 없음 (프리필만)     → 기본반 · sequences_per_s (출력값 그대로)
#   4. 검증 통과 확인 — "Validation: PASSED!" 가 없으면 제출하지 않음
#   5. aps-score 로 대시보드에 전송 (실행 로그 첨부)
#
# -v(검증)는 항상 자동으로 붙습니다. 정답과 다른 결과가 리더보드에 오르면
# 최적화 경쟁 자체가 무의미해지기 때문입니다. 검증은 측정 구간이 끝난 뒤에
# 실행되므로 기록에는 영향이 없습니다.
#
# 인증: 실습 서버에서 실행하면 리눅스 사용자명($USER = aps##)이 곧 신원입니다.
#       별도 키가 필요 없습니다.

set -uo pipefail

DASHBOARD="${APS_SERVER:-http://100.68.169.84:8080}"
PROJECT_DIR="${APS_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ORIG_ARGS=("$@")

NOTE=""
DRY_RUN=0
FROM_LOG=""
SELF_UPDATE=1
AS_USER=""
ADMIN_KEY=""
RUN_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --note)     NOTE="${2:-}"; shift 2 ;;
    --note=*)   NOTE="${1#*=}"; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --from-log) FROM_LOG="${2:-}"; shift 2 ;;
    --from-log=*) FROM_LOG="${1#*=}"; shift ;;
    --server)   DASHBOARD="${2:-}"; shift 2 ;;
    --server=*) DASHBOARD="${1#*=}"; shift ;;
    --no-update) SELF_UPDATE=0; shift ;;
    --as)       AS_USER="${2:-}"; shift 2 ;;
    --as=*)     AS_USER="${1#*=}"; shift ;;
    --admin-key)   ADMIN_KEY="${2:-}"; shift 2 ;;
    --admin-key=*) ADMIN_KEY="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          RUN_ARGS+=("$1"); shift ;;
  esac
done

# 기본값: 스켈레톤은 최적화 전이라 -n 1 이어야 slurm 시간 제한 안에 끝납니다.
[ ${#RUN_ARGS[@]} -eq 0 ] && RUN_ARGS=(-n 1)

# 검증(-v)은 선택이 아닙니다. main 이 타이머를 멈추고 Throughput 을 출력한 뒤에
# 정답과 비교하므로, 항상 켜도 측정값은 그대로입니다.
has_validate() {
  local a
  for a in "${RUN_ARGS[@]}"; do
    case "$a" in
      --validate) return 0 ;;
      --*)        ;;                  # 다른 롱옵션은 무시
      -*v*)       return 0 ;;         # -v, -dv 같은 묶음 표기까지
    esac
  done
  return 1
}
has_validate || RUN_ARGS+=(-v)

cd "$PROJECT_DIR" || { echo "프로젝트 폴더를 찾을 수 없습니다: $PROJECT_DIR" >&2; exit 1; }

if [ -t 1 ]; then C_OK=$'\033[92m'; C_ERR=$'\033[91m'; C_DIM=$'\033[90m'; C_HI=$'\033[96m'; C_END=$'\033[0m'
else C_OK=; C_ERR=; C_DIM=; C_HI=; C_END=; fi

# --- 자기 갱신 ---------------------------------------------------------------
# 참가자는 이 파일을 한 번 받아 두고 계속 씁니다. 판정 규칙이 바뀌었는데 손에 든
# 사본이 옛것이면, 서버는 거부만 하고 참가자는 이유를 알기 어렵습니다.
# 그래서 매 실행마다 최신본을 확인하고 달라졌으면 교체한 뒤 다시 실행합니다.
#
# 임시 파일을 같은 폴더에 만드는 것이 중요합니다. mv 가 같은 파일시스템 안의
# rename 이 되어야 새 inode 로 원자 교체가 되고, 지금 이 스크립트를 읽고 있는
# bash 는 옛 inode 를 그대로 붙들고 안전하게 끝까지 실행합니다.
SELF_DIR="$(dirname "$SELF")"
if [ "$SELF_UPDATE" = 1 ] && [ -z "${APS_SUBMIT_UPDATED:-}" ] \
   && [ -w "$SELF" ] && [ -w "$SELF_DIR" ]; then
  NEW="$(mktemp "$SELF_DIR/.submit.sh.XXXXXX" 2>/dev/null)"
  if [ -n "${NEW:-}" ]; then
    if curl -sf --max-time 10 "$DASHBOARD/submit.sh" -o "$NEW" \
       && [ "$(stat -c%s "$NEW" 2>/dev/null || echo 0)" -gt 2000 ] \
       && head -1 "$NEW" | grep -q '^#!/usr/bin/env bash' \
       && grep -q 'aps-score' "$NEW" \
       && ! cmp -s "$NEW" "$SELF"; then
      chmod +x "$NEW"
      mv -f "$NEW" "$SELF"
      echo "${C_DIM}submit.sh 를 최신본으로 갱신했습니다 — 다시 실행합니다.${C_END}"
      APS_SUBMIT_UPDATED=1 exec "$SELF" "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"
    fi
    rm -f "$NEW"
  fi
fi

if [ -n "$FROM_LOG" ]; then
  [ -r "$FROM_LOG" ] || { echo "로그를 읽을 수 없습니다: $FROM_LOG" >&2; exit 1; }
  LOG="$FROM_LOG"
  echo "${C_DIM}로그에서 결과를 읽습니다: $LOG${C_END}"
else
  [ -x ./run.sh ] || { echo "./run.sh 가 없습니다 (현재 위치: $PWD)" >&2; exit 1; }
  LOG="$(mktemp "${TMPDIR:-/tmp}/aps-run.XXXXXX.log")"
  trap 'rm -f "$LOG"' EXIT
  echo "${C_DIM}\$ ./run.sh ${RUN_ARGS[*]}${C_END}"
  ./run.sh "${RUN_ARGS[@]}" 2>&1 | tee "$LOG"
fi

# --- 결과 파싱 --------------------------------------------------------------
LINE="$(grep -E '^Throughput:' "$LOG" | tail -1)"
if [ -z "$LINE" ]; then
  echo >&2
  if grep -q 'DUE TO TIME LIMIT' "$LOG"; then
    echo "${C_ERR}✗ slurm 시간 제한에 걸려 결과가 나오지 않았습니다.${C_END}" >&2
    echo "  -n 값을 줄이거나(-n 1), 최적화 후 다시 시도하세요." >&2
  elif grep -qE 'CANCELLED|Terminated' "$LOG"; then
    echo "${C_ERR}✗ 작업이 중단되어 결과가 나오지 않았습니다.${C_END}" >&2
    echo "  squeue 로 큐 상태를 확인하고 다시 실행하세요." >&2
  else
    echo "${C_ERR}✗ 'Throughput:' 줄을 찾지 못했습니다. 실행이 실패한 것 같습니다.${C_END}" >&2
  fi
  exit 1
fi

VALUE="$(sed -E 's/^Throughput:[[:space:]]*([0-9.eE+-]+).*/\1/' <<<"$LINE")"
UNIT="$(sed -E 's/.*\(([^)]*)\).*/\1/' <<<"$LINE")"
ELAPSED="$(grep -E '^Elapsed time:' "$LOG" | tail -1 | sed -E 's/^Elapsed time:[[:space:]]*([0-9.eE+-]+).*/\1/')"
NSEQ="$(grep -E '^ Number of sequences:' "$LOG" | tail -1 | grep -oE '[0-9]+' | tail -1)"

# --- 수강반 판정 -------------------------------------------------------------
# 기본반은 프리필까지, 고급반은 8토큰 디코드까지 수행합니다. 프로그램은 두 경우 모두
# sequences/sec 을 출력하므로, 디코드일 때만 시퀀스당 토큰 수를 곱해 tok/s 로 바꿉니다.
DECODE_TOKENS="$(grep -oE 'Max decode tokens:[[:space:]]*[0-9]+' "$LOG" | grep -oE '[0-9]+' | tail -1)"
[ -z "${DECODE_TOKENS:-}" ] && DECODE_TOKENS="$(grep -oE 'MAX_DECODE_TOKENS[^0-9]*([0-9]+)' include/config.h 2>/dev/null | grep -oE '[0-9]+' | tail -1)"
: "${DECODE_TOKENS:=8}"

DECODE_HINT="-d 옵션 없음 · 프리필"
grep -qE '^ Decode: ON' "$LOG" && DECODE_HINT="-d 옵션 있음 · 디코드"

if grep -qE '^ Decode: ON' "$LOG"; then
  TRACK="고급반"
  METRIC="tokens_per_s"
  SUBMIT_VALUE="$(awk -v v="$VALUE" -v k="$DECODE_TOKENS" 'BEGIN{printf "%.6f", v*k}')"
  UNIT_LABEL="tok/s"
  DERIVED="${VALUE} seq/s × ${DECODE_TOKENS} tokens (디코드)"
else
  TRACK="기본반"
  METRIC="sequences_per_s"
  SUBMIT_VALUE="$VALUE"
  UNIT_LABEL="seq/s"
  DERIVED="프리필만 — 프로그램 출력 그대로"
fi

# --- 검증 판정 ---------------------------------------------------------------
# 정답과 다른 출력을 낸 최적화는 아무리 빨라도 기록이 될 수 없습니다.
if grep -q 'Validation: PASSED!' "$LOG"; then
  VERDICT="${C_OK}PASSED${C_END}"
  VALIDATED=1
elif grep -q 'Validation: FAILED' "$LOG"; then
  VERDICT="${C_ERR}FAILED${C_END}"
  VALIDATED=0
else
  VERDICT="${C_ERR}검증 결과 없음${C_END}"
  VALIDATED=0
fi

echo
echo "${C_HI}측정 결과${C_END}"
echo "  수강반     : ${C_HI}${TRACK}${C_END}  ${C_DIM}(${DECODE_HINT})${C_END}"
echo "  검증       : ${VERDICT}"
echo "  Throughput : ${VALUE} (${UNIT})"
[ -n "${ELAPSED:-}" ] && echo "  Elapsed    : ${ELAPSED} sec${NSEQ:+  ·  sequences: ${NSEQ}}"
echo "  제출 값     : ${C_HI}${SUBMIT_VALUE} ${UNIT_LABEL}${C_END}  ${C_DIM}(${METRIC}${NSEQ:+ · n=$NSEQ})${C_END}"
echo "  ${C_DIM}${DERIVED}${C_END}"

if [ "$VALIDATED" != 1 ]; then
  echo >&2
  echo "${C_ERR}✗ 검증을 통과하지 못해 제출하지 않았습니다.${C_END}" >&2
  if grep -q 'Validation: FAILED' "$LOG"; then
    grep -E '^Validation' "$LOG" | tail -4 | sed 's/^/    /' >&2
    echo "  출력이 정답과 다릅니다. 커널을 고친 뒤 다시 실행하세요." >&2
  elif [ -n "$FROM_LOG" ]; then
    echo "  이 로그는 -v 없이 실행된 것 같습니다. ./submit.sh 로 다시 측정하세요." >&2
  else
    echo "  검증 단계까지 가지 못했습니다 (시간 제한 또는 중단). 위 로그를 확인하세요." >&2
  fi
  exit 1
fi

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "${C_DIM}--dry-run: 제출하지 않았습니다.${C_END}"
  exit 0
fi

# --- 제출 -------------------------------------------------------------------
# ~/bin 을 먼저 본다: 실습 서버에는 과제 제출용 aps-submit 이 따로 있어서
# 이름으로만 찾으면 엉뚱한 도구가 걸린다.
# 관리자가 시스템 경로에 깔아두었으면 그것을 쓴다(자동 다운로드 없음).
# 없으면 ~/bin 에 받아서 쓰고, 실행할 때마다 최신본으로 갱신한다(9KB).
APS_SCORE=""
for cand in /usr/local/bin/aps-score /sbin/aps-score /usr/sbin/aps-score "$(command -v aps-score 2>/dev/null || true)"; do
  [ -n "$cand" ] && [ -x "$cand" ] && { APS_SCORE="$cand"; break; }
done
if [ -z "$APS_SCORE" ]; then
  APS_SCORE="$HOME/bin/aps-score"
  mkdir -p "$HOME/bin"
  if curl -sf --max-time 10 "$DASHBOARD/aps-score" -o "$APS_SCORE.tmp"; then
    mv -f "$APS_SCORE.tmp" "$APS_SCORE"; chmod +x "$APS_SCORE"
  else
    rm -f "$APS_SCORE.tmp"
    if [ ! -x "$APS_SCORE" ]; then
      echo "${C_ERR}✗ aps-score 를 받지 못했습니다 ($DASHBOARD 에 연결할 수 없음)${C_END}" >&2
      exit 1
    fi
    echo "${C_DIM}대시보드에 연결할 수 없어 캐시된 aps-score 를 사용합니다.${C_END}"
  fi
fi

[ -z "$NOTE" ] && NOTE="run.sh ${RUN_ARGS[*]}"

"$APS_SCORE" \
  --value "$SUBMIT_VALUE" \
  --metric "$METRIC" \
  ${NSEQ:+--n "$NSEQ"} \
  --note "$NOTE" \
  --log "$LOG" \
  --server "$DASHBOARD" \
  --meta "run_args=${RUN_ARGS[*]}" \
  --meta "raw_throughput=${VALUE} ${UNIT}" \
  --meta "elapsed_s=${ELAPSED:-}" \
  --meta "num_sequences=${NSEQ:-}" \
  --meta "decode_tokens=${DECODE_TOKENS}" \
  --meta "track=${TRACK}" \
  --meta "validated=passed" \
  ${AS_USER:+--as "$AS_USER"} \
  ${ADMIN_KEY:+--admin-key "$ADMIN_KEY"}
