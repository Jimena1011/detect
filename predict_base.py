# Ultralytics YOLO 🚀, GPL-3.0 license
#predict
#Camara 1
#avg_spd
#postprocess
#self.data_path
import hydra
import torch
import argparse
import time
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from omegaconf import DictConfig
#importa yolo desde ultralytics save
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import psycopg2
from datetime import datetime

from omegaconf import ListConfig

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import os
import sys



# Configuración de la base de datos
conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="1606", port="5432")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS prueba1 (
    id SERIAL PRIMARY KEY,
    clase varchar(255),
    speed varchar(255),
    way varchar (255),
    fecha varchar(255),
    camara varchar(255)
);
""")
consulta = """INSERT INTO prueba1 (clase, speed, way, fecha, camara) VALUES (%s, %s, %s, %s, %s);"""


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}
speed_line_queue = {}

line = [(100, 550), (1250, 550)]
line_1 = [(890,430),(1080,430)]  
line_2 = [(640,800),(1200,800)]

SOURCE_FPS = None
def get_source_fps(src_path):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None

###########Parte de la homografica#########################
#Pixeles 
src = np.array([[890,430],
                [1080,430],
                [1200,600],
                [640,800],
                ], dtype=np.float32)  # Puntos en la imagen (pixeles)
L = 27.69  # Largo en metro
W = 6.68   # Ancho en metros 
dst = np.array([[0.0,0],
                [L,0.0],
                [L,W],
                [0.0,W],
                ], dtype=np.float32)  # Puntos en el mundo real (metros)
H = cv2.getPerspectiveTransform(src, dst)  
def pix_to_world(pt_xy):
    P = np.array([[pt_xy]], dtype=np.float32)
    Wp = cv2.perspectiveTransform(P, H)[0,0]
    return float(Wp[0]), float(Wp[1])
speed_state = {}   # id -> {"p": (X,Y), "f": frame_idx}
speed_ema   = {}   # id -> EMA de km/h
EMA_ALPHA   = 0.3  # suavizado (0.2–0.4 suele ir bien)


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    #linera anterior 
    #base = Path(__file__).resolve().parent  # carpeta donde está predict.py
    yaml_path = base / "deep_sort_pytorch" / "configs" / "deep_sort.yaml"
    cfg_deep.merge_from_file(str(yaml_path))
    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                            min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=10,
                            #max_age=cfg_deep.DEEPSORT.MAX_AGE,
                            n_init=cfg_deep.DEEPSORT.N_INIT,
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True
                            )
################################################################################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "Sur"
    elif point1[1] < point2[1]:
        direction_str += "Norte"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
#def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0,0), frame_idx=None, fps=None):
    cv2.line(img, line[0], line[1], (46,162,112), 3)
    cv2.line(img, line_1[0], line_1[1], (46,162,112), 3)
    cv2.line(img, line_2[0], line_2[1], (46,162,112), 3)


    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)
        speed_line_queue.pop(key, None)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2)))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0
        if identities is not None:
            current_ids = set(int(x) for x in identities)
            for k in list(speed_state.keys()):
                if k not in current_ids:
                    speed_state.pop(k, None)
                    speed_ema.pop(k, None)


        # --- Asegura estructuras antes de usarlas ---
        if id not in data_deque:
            data_deque[id] = deque(maxlen=16)
        if id not in speed_line_queue:
            speed_line_queue[id] = []

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = f"{id}:{obj_name}"

        # calcula velocidad usando homografía
        now = None #ya no se usa time.time ()
        world_pt = pix_to_world(center)

        prev = speed_state.get(id)

        if prev is not None and fps and fps > 0 and ("f" in prev):
            df = max(1, frame_idx - prev["f"])  # delta de frames (>=1)
            dt = df / fps                       # segundos de video reales
            dx = world_pt[0] - prev["p"][0]
            dy = world_pt[1] - prev["p"][1]
            d_m = math.hypot(dx, dy)
            v_kmh_inst = (d_m / dt) * 3.6
            v_prev = speed_ema.get(id, v_kmh_inst)
            v_kmh = EMA_ALPHA * v_kmh_inst + (1.0 - EMA_ALPHA) * v_prev
            v_kmh = max(0.0, min(v_kmh, 300.0))
            speed_ema[id] = v_kmh
            object_speed = v_kmh
        else:
            object_speed = 0.0

            # actualiza estado
        speed_state[id] = {"p": world_pt, "f": frame_idx}

        # (sigue igual) acumulas para promedio mostrado
        speed_line_queue[id].append(object_speed)
        # add center to buffer (para dirección / trails)
        data_deque[id].appendleft(center)

        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "Sur" in direction:
                print(f"Intentando insertar: {obj_name}, {object_speed}, Sur")
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
        #para insertar los datos en la base de datos
                try: 
                    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    camara = os.getenv("CAMARA_NAME", "Camara_defecto")
                    cur.execute(consulta, (obj_name, str(object_speed), "Sur",fecha, camara))
                    #conn.commit()
                    print("Datos insertados en la base de datos.")
                except Exception as e:
                    print("Error al insertar datos en la base de datos:", e)

              if "Norte" in direction:
                print("Intentando insertar Norte", obj_name, object_speed)
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
                try: 
                    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    camara = os.getenv("CAMARA_NAME","Camara_defecto")
                    cur.execute(consulta, (obj_name, str(object_speed), "Norte",fecha, camara))
                    #conn.commit()
                    print("Datos insertados en la base de datos.")
                except Exception as e:
                    print("Error al insertar datos en la base de datos:", e)

        try:
            avg_spd = sum(speed_line_queue[id]) / max(1, len(speed_line_queue[id]))
            label = f"{label} {avg_spd:.1f} km/h"
        except Exception:
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)
        #    draw trail
        for i in range(1, len(data_deque[id])):
            #check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            #generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            #draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
    #4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
              cnt_str = str(key) + ":" +str(value)
              cv2.line(img, (width - 500, 90), (width,90), [85,45,255], 30)
              cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 95), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
              cv2.line(img, (width - 250, 125 + (idx*40)), (width - 50, 125 + (idx*40)), [85, 45, 255], 30)
              cv2.putText(img, cnt_str, (width - 250, 125 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
              cnt_str1 = str(key) + ":" +str(value)
              cv2.line(img, (20,25), (500,25), [85,45,255], 40)
              cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
              cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
              cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    return img

#La clase especial para la integracion con el Deepsort
class DetectionPredictor(BasePredictor):
    CLASS_ID = [0, 1, 2, 3, 5, 7]


    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        #print("¿CUDA disponible?:", torch.cuda.is_available())
        return img
    #Aqui se hace el precesaieto de los resultados
    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf, #Umbral de confianza
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds
    tiempo_inicio = time.time()  # Definir UNA SOLA VEZ al inicio imshow
    def write_results(self, idx, preds, batch, *args, **kwargs):
        t0 = time.time()
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame_idx = self.dataset.count
        else:
            frame_idx = getattr(self.dataset, 'frame', 0)
            
        fps = getattr(self.dataset, 'fps', SOURCE_FPS)
       # self.data_path = p
       # save_path = str(self.save_dir / p.name)  # im.jpg
       # self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            if int(cls) not in self.CLASS_ID:
                continue  # ignora clases no deseadas
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        if len(xywh_bboxs) == 0:
            outputs = np.empty((0, 5))
        else:
            xywhs  = torch.tensor(xywh_bboxs, dtype=torch.float32)
            confss = torch.tensor(confs,      dtype=torch.float32)
            outputs = deepsort.update(xywhs, confss, oids, im0)
          

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, offset=(0,0), frame_idx=frame_idx, fps=fps)
            #draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
            #tiempo de procesamiento
        t1 = time.time()  # <-- FINAL TIMER
        frame_time = max(t1 - t0, 1e-6)
        fps = 1 / frame_time
        tiempo_transcurrido =t1 - self.tiempo_inicio  # <-- Acceso a la variable global
        # Ahora sí, dibuja el texto sobre el frame
        #cv2.putText(im0, f"Frame time: {frame_time:.3f}s", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50,255,50), 3)
        cv2.putText(im0, f"FPS: {fps:.1f}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50,255,50), 3)
        #cv2.putText(im0, f"Tiempo total: {tiempo_transcurrido:.1f}s", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,50,50), 3)
        cv2.imshow("Seguimiento y conteo", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # presiona 'q' para salir
           print("Proceso detenido por el usuario.")
           cv2.destroyAllWindows()
           exit()

        return log_string

#@hydra.main(version_base=None, config_path=str(ROOT / "yolo" / "cfg"), config_name="default")
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.conf = 0.50
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    src_env = os.getenv("VIDEO_SOURCE")

    if src_env and str(src_env).strip():
        cfg.source = src_env
    else:
        cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    global SOURCE_FPS
    if isinstance(cfg.source, (str, os.PathLike)) and os.path.exists(str(cfg.source)):
        SOURCE_FPS = get_source_fps(str(cfg.source))
    else:
        SOURCE_FPS = float(os.getenv("CAM_FPS", 30.0))
    print("FPS de la fuente:", SOURCE_FPS)

    predictor = DetectionPredictor(cfg)
    predictor()
    cfg.show = True 

    print("¿CUDA disponible para torch?:", torch.cuda.is_available())
    
    # Verifica si el modelo YOLO está en CUDA (si tienes acceso al modelo, depende de la API)
    try:
        print("Dispositivo YOLO:", next(predictor.model.model.parameters()).device)
    except AttributeError:
        try:
            print("Dispositivo YOLO:", next(predictor.model.parameters()).device)
        except Exception as e:
            print(f"No se pudo acceder al dispositivo YOLO: {e}")
    

if __name__ == "__main__":
    predict()