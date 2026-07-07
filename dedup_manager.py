import hashlib
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DedupManager:
    """
    Gestor de deduplicación que:
    1. Evita enviar el mismo pick dos veces
    2. Limpia hashes de días anteriores automáticamente
    3. Resetea cada día para permitir nuevos picks diarios
    """

    def __init__(self, log_file):
        self.log_file = log_file
        self.sent_hashes = set()
        self.today = datetime.utcnow().strftime('%Y-%m-%d')
        self._load()

    def _load(self):
        """Carga los hashes del archivo y limpia los de días anteriores."""
        if not os.path.exists(self.log_file):
            self.sent_hashes = set()
            return

        try:
            with open(self.log_file, 'r') as f:
                lines = f.read().strip().splitlines()

            cleaned = set()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Formato nuevo: "YYYY-MM-DD|hash"
                if '|' in line:
                    date_part, hash_part = line.split('|', 1)
                    if date_part == self.today:
                        cleaned.add(hash_part)
                else:
                    # Formato legacy: conservar solo si es reciente
                    # (por compatibilidad, lo eliminamos al limpiar)
                    pass

            self.sent_hashes = cleaned

            # Reescribir el archivo limpio solo con hoy
            self._save()

            logger.info(
                f'📋 Dedup cargado: {len(self.sent_hashes)} picks '
                f'registrados para {self.today}'
            )

        except Exception as e:
            logger.error(f'❌ Error cargando dedup log: {e}')
            self.sent_hashes = set()

    def _save(self):
        """Guarda los hashes al archivo."""
        try:
            os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
            with open(self.log_file, 'w') as f:
                for h in self.sent_hashes:
                    f.write(f'{self.today}|{h}\n')
        except Exception as e:
            logger.error(f'❌ Error guardando dedup log: {e}')

    def generate_hash(self, *parts):
        """
        Genera un hash único a partir de los elementos dados.
        Ejemplo: generate_hash(home, away, time, market)
        """
        raw = '_'.join(str(p) for p in parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def ya_enviado(self, event_hash):
        """Verifica si un hash ya fue enviado."""
        return event_hash in self.sent_hashes

    def marcar_enviado(self, event_hash):
        """Marca un hash como enviado y persiste."""
        self.sent_hashes.add(event_hash)
        self._save()

    def cleanup_old_days(self):
        """Elimina registros de días anteriores del archivo."""
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, 'r') as f:
                lines = f.read().strip().splitlines()

            today_lines = []
            for line in lines:
                line = line.strip()
                if '|' in line:
                    date_part, _ = line.split('|', 1)
                    if date_part == self.today:
                        today_lines.append(line)

            with open(self.log_file, 'w') as f:
                for line in today_lines:
                    f.write(line + '\n')

            removed = len(lines) - len(today_lines)
            if removed > 0:
                logger.info(
                    f'🧹 Limpieza: {removed} registros antiguos eliminados'
                )

        except Exception as e:
            logger.error(f'❌ Error en limpieza: {e}')
