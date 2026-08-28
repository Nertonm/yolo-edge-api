#!/bin/bash
# scripts/deploy.sh
# Executa no Raspberry Pi via SSH pelo pipeline de CI/CD.
# Faz pull da nova imagem, reinicia o servico e valida o health check.
# Em caso de falha, reverte para a imagem anterior automaticamente.
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-${HOME}/yolo-edge-api}"
export DEPLOY_IMAGE="${DEPLOY_IMAGE:?DEPLOY_IMAGE must be set}"
HEALTH_URL="http://localhost:8000/health"
HEALTH_RETRIES=6
HEALTH_WAIT=10

echo "========================================"
echo " Deploy - $(date \"+%Y-%m-%d %H:%M:%S\")"
echo "========================================"
cd "$DEPLOY_PATH"

# Salva a imagem atual para possivel rollback
PREVIOUS=$(docker inspect yolo-api \
  --format '{{.Config.Image}}' 2>/dev/null || echo "none")
echo "[INFO] Imagem atual: $PREVIOUS"

echo "[1/4] Baixando nova imagem..."
docker compose pull

echo "[2/4] Iniciando nova versao..."
docker compose up -d

echo "[3/4] Aguardando health check ($((HEALTH_RETRIES * HEALTH_WAIT))s max)..."
SUCCESS=false
for i in $(seq 1 $HEALTH_RETRIES); do
  sleep $HEALTH_WAIT
  HEALTH_RESPONSE=$(curl -sf "$HEALTH_URL" 2>/dev/null || true)
  if [ -n "$HEALTH_RESPONSE" ] && printf "%s" "$HEALTH_RESPONSE" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status") == "ok" and d.get("model_loaded") is True'; then
    SUCCESS=true
    break
  fi
  echo "  Tentativa $i/$HEALTH_RETRIES falhou, aguardando..."
done

if [ "$SUCCESS" = true ]; then
  echo "[4/4] Health check OK"
  docker compose ps --status running --services | grep -qx yolo-api
  echo "[4/4] docker compose ps OK"
  NEW=$(docker inspect yolo-api --format '{{.Config.Image}}' 2>/dev/null)
  echo ""
  echo "[OK] Deploy bem-sucedido: $NEW"
  exit 0
else
  echo "[ERRO] Health check falhou apos $((HEALTH_RETRIES * HEALTH_WAIT))s"
  if [ "$PREVIOUS" != "none" ]; then
    echo "[ROLLBACK] Revertendo para: $PREVIOUS"
    docker compose down
    DEPLOY_IMAGE="$PREVIOUS" docker compose up -d
    echo "[ROLLBACK] Concluido. Servico restaurado."
  else
    echo "[AVISO] Sem imagem anterior para rollback."
  fi
  exit 1
fi
