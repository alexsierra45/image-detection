# Variables de entorno comunes (CHG-001): cachés y ajustes fuera de $HOME/.config y de /tmp del nodo.
# Los nodos de cómputo pueden no tener internet: pesos y fuentes se descargan antes con tools/fetch_weights.py.
_PROJ=/home/y404/y404034/image-detection
export YOLO_CONFIG_DIR=$_PROJ/cache/ultralytics      # settings.json de Ultralytics + Arial.ttf
export TORCH_HOME=$_PROJ/cache/torch                 # pesos de torchvision
export MPLCONFIGDIR=/home/y404/SCRATCH/y404034/tmp/mpl
export YOLO_OFFLINE=1                                # sin comprobaciones de versión/descargas en los nodos
export OPENCV_LOG_LEVEL=ERROR                        # las etiquetas GeoTIFF generan 5 avisos por tif (CHG-001)
unset _PROJ
