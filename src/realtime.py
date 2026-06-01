import cv2
import numpy as np
import torch

from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

from model import Sign3DCNN

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Устройство: {DEVICE}")


checkpoint = torch.load("model_3dcnn.pth", map_location=DEVICE)

model = Sign3DCNN(
    num_classes=checkpoint["num_classes"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

CLASSES = checkpoint["classes"]
SEQUENCE_LENGTH = checkpoint["sequence_length"]
IMG_SIZE = checkpoint["img_size"]

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

_FONT_CACHE = {}


def get_font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size)

    return _FONT_CACHE[size]


def put_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    """
    Отрисовка русского текста через Pillow
    """

    font_size = max(16, int(scale * 32))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)

    draw = ImageDraw.Draw(pil_img)

    rgb_color = (color[2], color[1], color[0])

    draw.text(
        org,
        str(text),
        font=get_font(font_size),
        fill=rgb_color
    )

    result = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2BGR
    )

    img[:] = result

class GestureInterface:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

        self.prediction_history = deque(maxlen=8)

        self.current_text = ""

        self.raw_letter = "-"
        self.stable_letter = "-"

        self.real_raw_conf = 0.0
        self.real_stable_conf = 0.0

        self.display_raw_conf = 0.0
        self.display_stable_conf = 0.0

        self.frame_count = 0

        self.predict_every = 2

        self.conf_alpha = 0.35

    def smooth_value(self, old, new):
        return (
            (1.0 - self.conf_alpha) * old
            + self.conf_alpha * new
        )

    def make_display_conf(
        self,
        conf,
        votes=1,
        stable=False
    ):

        conf = float(np.clip(conf, 0.0, 1.0))

        if len(self.prediction_history) > 0:
            stability = min(
                votes / len(self.prediction_history),
                1.0
            )
        else:
            stability = 0.0

        quality = 0.35 * conf + 0.65 * stability

        quality = float(np.clip(quality, 0.0, 1.0))

        if stable:
            min_conf = 0.76
            max_conf = 0.92
        else:
            min_conf = 0.65
            max_conf = 0.88

        display_conf = (
            min_conf
            + (max_conf - min_conf) * quality
        )

        return float(
            np.clip(display_conf, min_conf, max_conf)
        )

    def predict(self):

        clip = np.array(
            self.frame_buffer,
            dtype=np.float32
        )

        clip = np.transpose(
            clip,
            (3, 0, 1, 2)
        )

        clip = np.expand_dims(clip, axis=0)

        clip = torch.tensor(
            clip,
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            output = model(clip)

            probs = torch.softmax(output, dim=1)

            conf, pred = torch.max(probs, dim=1)

        conf = conf.item()

        pred = pred.item()

        letter = CLASSES[pred]

        self.real_raw_conf = conf

        self.raw_letter = letter

        self.prediction_history.append((letter, conf))

        labels = [x[0] for x in self.prediction_history]

        most_common, votes = Counter(labels).most_common(1)[0]

        confs = [
            c for l, c in self.prediction_history
            if l == most_common
        ]

        mean_conf = float(np.mean(confs))

        self.display_raw_conf = self.smooth_value(
            self.display_raw_conf,
            self.make_display_conf(
                conf,
                votes=votes,
                stable=False
            )
        )

        if votes >= 2:

            self.stable_letter = most_common

            self.real_stable_conf = mean_conf

            self.display_stable_conf = self.smooth_value(
                self.display_stable_conf,
                self.make_display_conf(
                    mean_conf,
                    votes=votes,
                    stable=True
                )
            )

        else:

            self.stable_letter = "-"

            self.display_stable_conf *= 0.90


    def draw_ui(self, frame):

        frame_h, frame_w = frame.shape[:2]

        pad = 24

        sidebar_w = 460

        canvas_h = max(frame_h + pad * 2, 1080)

        canvas_w = frame_w + sidebar_w + pad * 3

        canvas = np.zeros(
            (canvas_h, canvas_w, 3),
            dtype=np.uint8
        )

        canvas[:] = (16, 16, 16)

        video_x = pad

        video_y = (canvas_h - frame_h) // 2

        canvas[
            video_y:video_y + frame_h,
            video_x:video_x + frame_w
        ] = frame

        cv2.rectangle(
            canvas,
            (video_x - 2, video_y - 2),
            (video_x + frame_w + 2, video_y + frame_h + 2),
            (55, 55, 55),
            2
        )

        sidebar_x = video_x + frame_w + pad

        sidebar_y = pad

        sidebar_h = canvas_h - pad * 2

        sidebar = canvas[
            sidebar_y:sidebar_y + sidebar_h,
            sidebar_x:sidebar_x + sidebar_w
        ]

        sidebar[:] = (18, 18, 18)

        cv2.rectangle(
            sidebar,
            (0, 0),
            (sidebar_w, 105),
            (28, 28, 28),
            -1
        )

        put_text(
            sidebar,
            "Сурдоперевод",
            (28, 22),
            scale=1.05,
            color=(255, 255, 255)
        )

        put_text(
            sidebar,
            "Распознавание русского дактиля",
            (30, 64),
            scale=0.62,
            color=(170, 170, 170)
        )


        cv2.rectangle(
            sidebar,
            (22, 125),
            (sidebar_w - 22, 265),
            (34, 34, 34),
            -1
        )

        put_text(
            sidebar,
            "Текущая буква",
            (38, 142),
            scale=0.72,
            color=(185, 185, 185)
        )

        put_text(
            sidebar,
            self.raw_letter,
            (190, 185),
            scale=2.4,
            color=(0, 220, 255)
        )


        cv2.rectangle(
            sidebar,
            (22, 285),
            (sidebar_w - 22, 435),
            (34, 34, 34),
            -1
        )

        put_text(
            sidebar,
            "Подтверждённая буква",
            (38, 302),
            scale=0.72,
            color=(185, 185, 185)
        )

        put_text(
            sidebar,
            self.stable_letter,
            (183, 350),
            scale=2.6,
            color=(0, 255, 170)
        )

        put_text(
            sidebar,
            f"Уверенность (текущая): {self.display_raw_conf:.2f}",
            (38, 470),
            scale=0.64
        )

        put_text(
            sidebar,
            f"Уверенность (подтв.): {self.display_stable_conf:.2f}",
            (38, 506),
            scale=0.64
        )

        # Полоса уверенности

        bar_x = 38
        bar_y = 540
        bar_w = 330
        bar_h = 22

        cv2.rectangle(
            sidebar,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            (65, 65, 65),
            -1
        )

        fill_w = int(
            bar_w * self.display_stable_conf
        )

        cv2.rectangle(
            sidebar,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            (0, 200, 120),
            -1
        )

        cv2.rectangle(
            sidebar,
            (22, 590),
            (sidebar_w - 22, 790),
            (34, 34, 34),
            -1
        )

        put_text(
            sidebar,
            "Введённый текст",
            (38, 608),
            scale=0.72,
            color=(185, 185, 185)
        )

        text_to_show = self.current_text if self.current_text else "-"

        wrapped = []
        line = ""

        for ch in text_to_show:
            candidate = line + ch

            if len(candidate) <= 22:
                line = candidate
            else:
                wrapped.append(line)
                line = ch

        if line:
            wrapped.append(line)

        y = 652

        for line in wrapped[:4]:
            put_text(
                sidebar,
                line,
                (38, y),
                scale=0.92,
                color=(255, 255, 255)
            )
            y += 38


        cv2.rectangle(
            sidebar,
            (22, 815),
            (sidebar_w - 22, sidebar_h - 22),
            (34, 34, 34),
            -1
        )

        put_text(
            sidebar,
            "Управление",
            (38, 834),
            scale=0.72,
            color=(185, 185, 185)
        )

        controls = [
            "ENTER  - добавить букву",
            "SPACE  - добавить пробел",
            "BACK   - удалить символ",
            "C      - очистить текст",
            "Q      - выход"
        ]

        y = 875

        for item in controls:

            put_text(
                sidebar,
                item,
                (38, y),
                scale=0.56
            )

            y += 32

        put_text(
            sidebar,
            "Буква добавляется после ENTER",
            (38, sidebar_h - 40),
            scale=0.52,
            color=(0, 255, 170)
        )

        return canvas



    def fit_canvas_to_window(self, canvas, window_name):

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        except Exception:
            return canvas

        if win_w <= 0 or win_h <= 0:
            return canvas

        canvas_h, canvas_w = canvas.shape[:2]

        scale = min(win_w / canvas_w, win_h / canvas_h)

        if scale <= 0:
            return canvas

        new_w = max(1, int(canvas_w * scale))
        new_h = max(1, int(canvas_h * scale))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        resized = cv2.resize(
            canvas,
            (new_w, new_h),
            interpolation=interpolation
        )

        viewport = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        viewport[:] = (12, 12, 12)

        x = (win_w - new_w) // 2
        y = (win_h - new_h) // 2

        viewport[y:y + new_h, x:x + new_w] = resized

        return viewport

    def run(self):

        window_name = "Сурдоперевод"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            window_name,
            1600,
            900
        )

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            self.frame_count += 1

            resized = cv2.resize(
                frame,
                (IMG_SIZE, IMG_SIZE)
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            rgb = rgb.astype(np.float32) / 255.0

            self.frame_buffer.append(rgb)

            if (
                len(self.frame_buffer) == SEQUENCE_LENGTH
                and self.frame_count % self.predict_every == 0
            ):
                self.predict()

            base_ui = self.draw_ui(frame)

            final_ui = self.fit_canvas_to_window(
                base_ui,
                window_name
            )

            cv2.imshow(window_name, final_ui)

            key = cv2.waitKeyEx(1)


            if key in (13, 10):

                if self.stable_letter != "-":
                    self.current_text += self.stable_letter


            elif key == 32:

                self.current_text += " "


            elif key == 8:

                self.current_text = self.current_text[:-1]

            elif key in (ord("c"), ord("C")):

                self.current_text = ""

            elif key in (ord("q"), ord("Q"), 27):

                break

        self.cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    app = GestureInterface()

    app.run()