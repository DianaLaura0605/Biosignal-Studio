"""
BioSignal Studio — Backend Principal
=====================================
Plataforma de adquisición, procesamiento y análisis de señales biomédicas
para aplicaciones de control de prótesis biónicas.

Autor: BioSignal Studio
Stack: FastAPI + NumPy + SciPy + WebSockets
Objetivo: Demostración de procesamiento de señales EMG/ECG/EEG en tiempo real
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from app.api import signals, analysis, prosthetics, sessions
from app.core.config import settings
from app.core.connection_manager import ConnectionManager

# ─── Configuración de logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("biosignal_studio")

# ─── Gestor de conexiones WebSocket (singleton) ───────────────────────────────
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de la aplicación.
    Se ejecuta al iniciar y al cerrar el servidor.
    Aquí se inicializarían conexiones a DB, caches, etc.
    """
    logger.info("🧬 BioSignal Studio iniciando...")
    logger.info(f"   Versión: {settings.VERSION}")
    logger.info(f"   Entorno: {settings.ENVIRONMENT}")
    yield
    # Cleanup al cerrar
    logger.info("🛑 BioSignal Studio cerrando conexiones...")
    await manager.disconnect_all()


# ─── Instancia principal de la app ────────────────────────────────────────────
app = FastAPI(
    title="BioSignal Studio API",
    description="""
    ## API para procesamiento de señales biomédicas

    Endpoints para:
    - **Señales**: adquisición y simulación de EMG, ECG, EEG
    - **Análisis**: FFT, filtrado, detección de patrones
    - **Prótesis**: clasificación de gestos y control en tiempo real
    - **Sesiones**: grabación y reproducción de sesiones clínicas
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ─── CORS — permite al frontend React conectarse ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers — cada módulo tiene su propio router ─────────────────────────────
app.include_router(signals.router,    prefix="/api/signals",    tags=["Señales Biomédicas"])
app.include_router(analysis.router,   prefix="/api/analysis",   tags=["Análisis DSP"])
app.include_router(prosthetics.router, prefix="/api/prosthetics", tags=["Control Prótesis"])
app.include_router(sessions.router,   prefix="/api/sessions",   tags=["Sesiones Clínicas"])


# ─── WebSocket — streaming de señales en tiempo real ──────────────────────────
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Canal WebSocket principal.
    Cada cliente se conecta con un ID único y recibe streaming
    de datos de señal en tiempo real a ~250 Hz (4ms por muestra).
    """
    await manager.connect(websocket, client_id)
    logger.info(f"Cliente conectado: {client_id}")

    try:
        while True:
            # Esperar mensajes del cliente (comandos de control)
            data = await websocket.receive_text()
            message = json.loads(data)

            # Procesar comandos: start_stream, stop_stream, change_signal, etc.
            await manager.handle_command(websocket, client_id, message)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Cliente desconectado: {client_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket {client_id}: {e}")
        manager.disconnect(client_id)


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verifica que el servidor esté operacional."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "active_connections": manager.active_connections_count,
    }


@app.get("/", tags=["Sistema"])
async def root():
    """Raíz de la API — redirige a documentación."""
    return JSONResponse({
        "message": "BioSignal Studio API",
        "docs": "/api/docs",
        "version": settings.VERSION,
    })
