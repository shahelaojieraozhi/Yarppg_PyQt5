# yarppg

## main.py



```python
def main():
    args = _mainparser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    roi_detector = get_detector(args)

    digital_lowpass = get_butterworth_filter(30, 1.5)
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=300,
                           filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

    processor = get_processor(args)

    cutoff = parse_frequencies(args.bandpass)
    if cutoff is not None:
        digital_bandpass = get_butterworth_filter(30, cutoff, "bandpass")
        processor = FilteredProcessor(processor, digital_bandpass)

    rppg = RPPG(roi_detector=roi_detector,
                video=args.video,
                hr_calculator=hr_calc,
                parent=app,
                )
    rppg.add_processor(processor)
    for c in "rgb":
        rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))

    if args.savepath:
        rppg.output_filename = args.savepath

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     blur_roi=args.blur,
                     )
    for i in range(3):
        win.set_pen(index=i + 1, color="rgb"[i], width=1)

    return win.execute()


if __name__ == "__main__":
    sys.exit(main())
```



mainwindow.py

```python

def update_hr(self, hr):
    self.hr_label.setText("Heart rate: {:5.1f} beat/min".format(hr))
```







region_of_interest.py

```python
def draw_roi(self, img, color=(255, 0, 0), thickness=3):
    if self.is_empty():
        return

    if self.is_rectangular():
        p1, p2 = self.get_bounding_box(as_corners=True)
        cv2.rectangle(img, p1, p2, color, thickness)
    else:
        cv2.drawContours(img, self._contours, 0, color=color,
                         thickness=thickness)
```













```python
def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    return DigitalFilter(ba[0], ba[1])
```







选择参数：

```python
_mainparser = argparse.ArgumentParser(description="Use your Webcam to measure"
                                                  "your heart rate")
_mainparser.add_argument("--detector", default="facemesh", type=str,
                         choices=["facemesh", "caffe-dnn", "haar", "full"],
                         help="ROI (face) detector")
_mainparser.add_argument("--processor", default="LiCvpr",
                         choices=["LiCvpr", "Pos", "Chrom"],
                         help=("Processor translating ROI to pulse signal. "
                               "LiCvpr currently only returns mean green value"))
_mainparser.add_argument("--winsize", default=32, type=int,
                         help="Window sized used in some processors")
_mainparser.add_argument("--bandpass", type=str, default="0.5,2",
                         help="bandpass frequencies for processor output")
_mainparser.add_argument("--blobsize", default=150, type=int,
                         help="quadratic blob size of DNN Face Detector")
_mainparser.add_argument("--draw-facemark", action="store_true",
                         help="draw landmarks when using facemesh detector")
_mainparser.add_argument("--blur", default=-1, type=int,
                         help="pixelation size of detected ROI")
_mainparser.add_argument("--video", default=0, type=int,
                         help="video input device number")
_mainparser.add_argument("--savepath", default="", type=str,
                         help="store generated signals as data frame")
```

默认：

```python
get_detector:default="facemesh"
get_processor:default="LiCvpr"
    
--winsize:default=32
--bandpass：default="0.5,2"
--blobsize ：default=150
--draw-facemark ：action="store_true"
--blur：default=-1  "pixelation size of detected ROI"
--video  default=0	"video input device number"
--savepath  default=""  "store generated signals as data frame"
```











**获取信号的检测器**

```python
def get_detector(args):
    name = args.detector.lower()

    if name == "full":
        return NoDetector()
    elif name == "facemesh":
        return FaceMeshDetector(draw_landmarks=args.draw_facemark)
    elif name == "caffe-dnn":
        return CaffeDNNFaceDetector(blob_size=args.blobsize)
    elif name == "haar":
        return HaarCascadeDetector()

    raise NotImplementedError(f"detector {args.detector!r} not recognized.")
```











**使用谁的处理方法：**

```python
def get_processor(args):
    name = args.processor.lower()
    if name == "licvpr":
        return LiCvprProcessor()
    elif name == "pos":
        return PosProcessor(winsize=args.winsize)
    elif name == "chrom":
        return ChromProcessor(winsize=args.winsize, method="xovery")
```



？

```python
def parse_frequencies(s):
    """
    Rudimentary parser of frequency string expected as 'f1,f2' (e.g. '0.4,2')
    """
    if s == "none":
        return None
    return list(map(float, s.split(",")))
```





**main.py**

```python
def main():
    args = _mainparser.parse_args(sys.argv[1:])
    app = QApplication(sys.argv)

    roi_detector = get_detector(args)

    digital_lowpass = get_butterworth_filter(30, 1.5)
    hr_calc = HRCalculator(parent=app, update_interval=30, winsize=300,
                           filt_fun=lambda vs: [digital_lowpass(v) for v in vs])

    processor = get_processor(args)

    cutoff = parse_frequencies(args.bandpass)
    if cutoff is not None:
        digital_bandpass = get_butterworth_filter(30, cutoff, "bandpass")
        processor = FilteredProcessor(processor, digital_bandpass)

    rppg = RPPG(roi_detector=roi_detector,
                video=args.video,
                hr_calculator=hr_calc,
                parent=app,
                )
    rppg.add_processor(processor)
    for c in "rgb":
        rppg.add_processor(ColorMeanProcessor(channel=c, winsize=1))

    if args.savepath:
        rppg.output_filename = args.savepath

    win = MainWindow(app=app,
                     rppg=rppg,
                     winsize=(1000, 400),
                     legend=True,
                     graphwin=300,
                     blur_roi=args.blur,
                     )
    for i in range(3):
        win.set_pen(index=i + 1, color="rgb"[i], width=1)

    return win.execute()
```

**rppg.py**

```python
class RPPG(QObject):
    new_update = pyqtSignal(float)
    _dummy_signal = pyqtSignal(float)

    def __init__(self, roi_detector, parent=None, video=0,
                 hr_calculator=None):
        QObject.__init__(self, parent)
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector

        self._set_camera(video)

        self._dts = []
        self.last_update = datetime.now()

        self.output_frame = None
        self.hr_calculator = hr_calculator

        if self.hr_calculator is not None:
            self.new_hr = self.hr_calculator.new_hr
        else:
            self.new_hr = self._dummy_signal

        self.output_filename = None

    def _set_camera(self, video):
        self._cam = Camera(video=video, parent=self)
        self._cam.new_frame.connect(self.frame_received)

    def add_processor(self, processor):
        self._processors.append(processor)

    def frame_received(self, frame):
        self.output_frame = frame
        self.roi = self._roi_detector(frame)

        for processor in self._processors:
            processor(self.roi)

        if self.hr_calculator is not None:
            self.hr_calculator.update(self)

        dt = self._update_time()
        self.new_update.emit(dt)

    def _update_time(self):
        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self._dts.append(dt)

        return dt

    def get_vs(self, n=None):
        for processor in self._processors:
            if n is None:
                yield np.array(processor.vs, copy=True)
            else:
                yield np.array(processor.vs[-n:], copy=True)

    def get_ts(self, n=None):
        if n is None:
            dts = self._dts
        else:
            dts = self._dts[-n:]
        return np.cumsum(dts)

    def get_fps(self, n=5):
        return 1/np.mean(self._dts[-n:])

    def save_signals(self):
        path = pathlib.Path(self.output_filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_dataframe()
        write_dataframe(path)

    def get_dataframe(self):
        names = ["ts"] + ["p%d" % i for i in range(self.num_processors)]
        data = np.vstack((self.get_ts(),) + tuple(self.get_vs())).T

        return pd.DataFrame(data=data, columns=names)

    @property
    def num_processors(self):
        return len(self._processors)

    @property
    def processor_names(self):
        return [str(p) for p in self._processors]

    def start(self):
        self._cam.start()

    def finish(self):
        print("finishing up...")
        if self.output_filename is not None:
            self.save_signals()
        self._cam.stop()
```









filters.py

```python
import numpy as np
import scipy.signal


def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    return DigitalFilter(ba[0], ba[1])


class DigitalFilter:

    def __init__(self, b, a):
        self._bs = b
        self._as = a
        self._xs = [0]*len(b)
        self._ys = [0]*(len(a)-1)

    def process(self, x):
        if np.isnan(x):  # ignore nans, and return as is
            return x

        self._xs.insert(0, x)
        self._xs.pop()
        y = (np.dot(self._bs, self._xs) / self._as[0]
             - np.dot(self._as[1:], self._ys))
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)


if __name__ == "__main__":
    fs = 30
    x = np.arange(0, 10, 1.0/fs)
    y = np.sin(2*np.pi*x) + 0.2*np.random.normal(size=len(x))

    import pyqtgraph as pg
    app = pg.QtGui.QApplication([])
    p = pg.plot(title="test")
    p.plot(x, y)
    ba = scipy.signal.butter(2, 3/fs*2)
    yfilt = scipy.signal.lfilter(ba[0], ba[1], y)
    p.plot(x, yfilt, pen=(0, 3))

    myfilt = DigitalFilter(ba[0], ba[1])
    yfilt2 = [myfilt(v) for v in y]
    p.plot(x, yfilt2, pen=(1, 3))
    app.exec_()
```

直接运行：

![](E:\Picture_bin\2022_7\Snipaste_2022-07-20_18-33-36.jpg)































li_cvpr

```python
import numpy as np

from .processor import Processor


class LiCvprProcessor(Processor):
    def __init__(self, winsize=1):
        super().__init__()

        self.winsize = winsize

    def calculate(self, roi):
        r, g, b = self.spatial_pooling(roi)

        return g

    def __str__(self):
        if self.name is None:
            return f"LiCvprProcessor(winsize={self.winsize})"
        return self.name
```





processor,py

```python
def spatial_pooling(self, roi, append_rgb=False):
    r, g, b = roi.get_mean_rgb()

    if append_rgb:
        self._rs.append(r)
        self._gs.append(g)
        self._bs.append(b)

    return r, g, b
```

调用了region_of_interest.py里的get_mean_rgb

```python
def get_mean_rgb(self, background=False):
    mask = self._mask
    if background:
        if self._bgmask is None:
            raise ValueError("Background mask is not specified")
        mask = self._bgmask

    r, g, b, a = cv2.mean(self.rawimg, mask)
    return r, g, b
```





```python
class FilteredProcessor(Processor):
    def __init__(self, processor, filtfun):
        Processor.__init__(self)
        self._processor = processor
        self._filtfun = filtfun
        self.name = "Filtered" + str(processor)

    def calculate(self, roi):
        v = self._filtfun(self._processor.calculate(roi))
        return v
```





chrom.py

```python
import numpy as np

from .processor import Processor


class ChromProcessor(Processor):

    def __init__(self, winsize=45, method="xovery"):
        Processor.__init__(self)

        self.winsize = winsize
        self.method = method

        self._xs, self._ys = [], []
        self.xmean, self.ymean = 0, 0
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        r, g, b = self.spatial_pooling(roi_pixels, append_rgb=True)
        v = np.nan

        if self.method == "fixed":
            self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
            self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
            self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)
            rn = r / (self.rmean or 1.)
            gn = g / (self.gmean or 1.)
            bn = b / (self.bmean or 1.)
            self._xs.append(3*rn - 2*gn)
            self._ys.append(1.5*rn + gn - 1.5*bn)

            v = self._xs[-1] / (self._ys[-1] or 1.) - 1
        elif self.method == "xovery":
            self._xs.append(r - g)
            self._ys.append(0.5*r + 0.5*g - b)
            self.xmean = self.moving_average_update(self.xmean, self._xs, self.winsize)
            self.ymean = self.moving_average_update(self.ymean, self._ys, self.winsize)

            v = self.xmean / (self.ymean or 1) - 1

        return v

    def __str__(self):
        if self.name is None:
            return "ChromProcessor(winsize={},method={})".format(self.winsize,
                                                                 self.method)
        return self.name
```



















```python
from yarppg.rppg import RPPG
```



```python
class RPPG(QObject):
    new_update = pyqtSignal(float)
    _dummy_signal = pyqtSignal(float)

    def __init__(self, roi_detector, parent=None, video=0,
                 hr_calculator=None):
        QObject.__init__(self, parent)
        self.roi = None
        self._processors = []
        self._roi_detector = roi_detector

        self._set_camera(video)

        self._dts = []
        self.last_update = datetime.now()

        self.output_frame = None
        self.hr_calculator = hr_calculator

        if self.hr_calculator is not None:
            self.new_hr = self.hr_calculator.new_hr
        else:
            self.new_hr = self._dummy_signal

        self.output_filename = None

    def _set_camera(self, video):
        self._cam = Camera(video=video, parent=self)
        self._cam.new_frame.connect(self.frame_received)

    def add_processor(self, processor):
        self._processors.append(processor)

    def frame_received(self, frame):
        self.output_frame = frame
        self.roi = self._roi_detector(frame)

        for processor in self._processors:
            processor(self.roi)

        if self.hr_calculator is not None:
            self.hr_calculator.update(self)

        dt = self._update_time()
        self.new_update.emit(dt)

    def _update_time(self):
        dt = (datetime.now() - self.last_update).total_seconds()
        self.last_update = datetime.now()
        self._dts.append(dt)

        return dt

    def get_vs(self, n=None):
        for processor in self._processors:
            if n is None:
                yield np.array(processor.vs, copy=True)
            else:
                yield np.array(processor.vs[-n:], copy=True)

    def get_ts(self, n=None):
        if n is None:
            dts = self._dts
        else:
            dts = self._dts[-n:]
        return np.cumsum(dts)

    def get_fps(self, n=5):
        return 1/np.mean(self._dts[-n:])

    def save_signals(self):
        path = pathlib.Path(self.output_filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_dataframe()
        write_dataframe(path)

    def get_dataframe(self):
        names = ["ts"] + ["p%d" % i for i in range(self.num_processors)]
        data = np.vstack((self.get_ts(),) + tuple(self.get_vs())).T

        return pd.DataFrame(data=data, columns=names)

    @property
    def num_processors(self):
        return len(self._processors)

    @property
    def processor_names(self):
        return [str(p) for p in self._processors]

    def start(self):
        self._cam.start()

    def finish(self):
        print("finishing up...")
        if self.output_filename is not None:
            self.save_signals()
        self._cam.stop()
```





```python
from yarppg.rppg.processors import (ColorMeanProcessor, ChromProcessor,
                                    FilteredProcessor, PosProcessor, LiCvprProcessor)
```





提前导入的一个Processor，目前先认为它是个模板(processor.py的一个类)

```python
class Processor:
    def __init__(self):
        self.name = None
        self._rs = []
        self._gs = []
        self._bs = []

        self.vs = []

    def calculate(self, roi):
        return np.nan

    def __call__(self, roi):
        v = self.calculate(roi)
        self.vs.append(v)
        return v

    def spatial_pooling(self, roi, append_rgb=False):
        r, g, b = roi.get_mean_rgb()

        if append_rgb:
            self._rs.append(r)
            self._gs.append(g)
            self._bs.append(b)

        return r, g, b

    def __str__(self):
        if self.name is None:
            return "Processor"
        else:
            return self.name

    @staticmethod
    def moving_average_update(xold, xs, winsize):
        if len(xs) == 0:
            return np.nan
        '''
        n = len(xs)
        if n == 0:
            return 0
        if n < winsize:
            return sum(xs) / len(xs)
        return xold + (xs[-1] - xs[max(0, n - winsize)]) / min(n, winsize)
        '''
        return np.nanmean(xs[-winsize:])
```



```python
class ColorMeanProcessor(Processor):
    channel_dict = dict(r=0, g=1, b=2)
    channel_dict_reverse = {0: "r", 1: "g", 2: "b"}

    def __init__(self, channel="g", winsize=1):
        Processor.__init__(self)

        if channel not in self.channel_dict.keys():
            raise KeyError("channel has to be one of "
                           "{}".format(set(self.channel_dict.keys())))

        self.channel = self.channel_dict[channel]
        self.winsize = winsize
        self._tmp = []

    def calculate(self, roi_pixels):
        rgb = self.spatial_pooling(roi_pixels, append_rgb=False)
        self._tmp.append(rgb[self.channel])
        return self.moving_average_update(0, self._tmp, self.winsize)

    def __str__(self):
        if self.name is None:
            channel = self.channel_dict_reverse[self.channel]
            return "ColorMean(winsize={},c={})".format(self.winsize, channel)
        return self.name
```



```python
class ChromProcessor(Processor):

    def __init__(self, winsize=45, method="xovery"):
        Processor.__init__(self)

        self.winsize = winsize
        self.method = method

        self._xs, self._ys = [], []
        self.xmean, self.ymean = 0, 0
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        r, g, b = self.spatial_pooling(roi_pixels, append_rgb=True)
        v = np.nan

        if self.method == "fixed":
            self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
            self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
            self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)
            rn = r / (self.rmean or 1.)
            gn = g / (self.gmean or 1.)
            bn = b / (self.bmean or 1.)
            self._xs.append(3*rn - 2*gn)
            self._ys.append(1.5*rn + gn - 1.5*bn)

            v = self._xs[-1] / (self._ys[-1] or 1.) - 1
        elif self.method == "xovery":
            self._xs.append(r - g)
            self._ys.append(0.5*r + 0.5*g - b)
            self.xmean = self.moving_average_update(self.xmean, self._xs, self.winsize)
            self.ymean = self.moving_average_update(self.ymean, self._ys, self.winsize)

            v = self.xmean / (self.ymean or 1) - 1

        return v

    def __str__(self):
        if self.name is None:
            return "ChromProcessor(winsize={},method={})".format(self.winsize,
                                                                 self.method)
        return self.name
```



```python
class FilteredProcessor(Processor):
    def __init__(self, processor, filtfun):
        Processor.__init__(self)
        self._processor = processor
        self._filtfun = filtfun
        self.name = "Filtered" + str(processor)

    def calculate(self, roi):
        v = self._filtfun(self._processor.calculate(roi))
        return v
```





```python
class PosProcessor(Processor):
    def __init__(self, winsize=45):
        Processor.__init__(self)

        self.winsize = winsize

        self.hs = []
        self.rmean, self.gmean, self.bmean = 0, 0, 0

        self.n = 0

    def calculate(self, roi_pixels):
        self.n += 1
        self.spatial_pooling(roi_pixels, append_rgb=True)

        # spatial averaging
        self.rmean = self.moving_average_update(self.rmean, self._rs, self.winsize)
        self.gmean = self.moving_average_update(self.gmean, self._gs, self.winsize)
        self.bmean = self.moving_average_update(self.bmean, self._bs, self.winsize)

        if self.n >= self.winsize:
            # temporal normalization
            rn = np.divide(self._rs[-self.winsize:], self.rmean or 1.)
            gn = np.divide(self._gs[-self.winsize:], self.gmean or 1.)
            bn = np.divide(self._bs[-self.winsize:], self.bmean or 1.)

            # projection
            s1 = gn - bn
            s2 = -2*rn + gn + bn

            # tuning
            h = s1 + np.nanstd(s1) / np.nanstd(s2) * s2
            self.hs.append(0.)
            self.hs[-self.winsize:] = self.hs[-self.winsize:] + (h-np.nanmean(h))
            return self.hs[-self.winsize]
        self.hs.append(0)
        return 0

    def __str__(self):
        if self.name is None:
            return "PosProcessor(winsize={})".format(self.winsize)
        return self.name
```



```python
class LiCvprProcessor(Processor):
    def __init__(self, winsize=1):
        super().__init__()

        self.winsize = winsize

    def calculate(self, roi):
        r, g, b = self.spatial_pooling(roi)

        return g

    def __str__(self):
        if self.name is None:
            return f"LiCvprProcessor(winsize={self.winsize})"
        return self.name
```











计算心率：

```python
class HRCalculator(QObject):
    new_hr = pyqtSignal(float)

    def __init__(self, parent=None, update_interval=30, winsize=300,
                 filt_fun=None, hr_fun=None):
        QObject.__init__(self, parent)

        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.hr_fun = from_peaks
        if hr_fun is not None and callable(hr_fun):
            self.hr_fun = hr_fun

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            if self.filt_fun is not None and callable(self.filt_fun):
                vs = self.filt_fun(vs)
            self.new_hr.emit(self.hr_fun(vs, ts))
```









```python
from yarppg.rppg.roi.roi_detect import CaffeDNNFaceDetector, FaceMeshDetector, HaarCascadeDetector, NoDetector
```



```python
class NoDetector(ROIDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame):
        h, w = frame.shape[:2]
        return RegionOfInterest.from_rectangle(frame, (0, 0), (h, w))
```



```python
class CaffeDNNFaceDetector(ROIDetector):
    prototxt = resource_path / "deploy.prototxt"
    caffemodel = resource_path / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    color_mean = (128, 128, 128)

    def __init__(self, prototxt=None, caffemodel=None,
                 blob_size=(300, 300),
                 min_confidence=0.3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        print(self.caffemodel)
        self.blob_size = blob_size
        self.min_confidence = min_confidence
        if prototxt is None:
            prototxt = self.prototxt
        if caffemodel is None:
            caffemodel = self.caffemodel
        self.model = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.blob_size, self.color_mean)
        self.model.setInput(blob)
        detections = self.model.forward()[0, 0, ...]
        for det in detections:
            if det[2] > self.min_confidence:
                x1, y1, x2, y2 = np.multiply(
                    det[3:7], (w, h, w, h)).astype(int)
                return RegionOfInterest.from_rectangle(frame, (x1, y1), (x2, y2))
        return RegionOfInterest(frame)
```



```python
class HaarCascadeDetector(ROIDetector):
    default_cascade = resource_path / "haarcascade_frontalface_default.xml"

    def __init__(self,
                 casc_file=None,
                 scale_factor=1.1,
                 min_neighbors=5,
                 min_size=(30, 30),
                 **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = self._get_classifier(casc_file)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray,
                                              scaleFactor=self.scale_factor,
                                              minNeighbors=self.min_neighbors,
                                              )  # minSize=self.min_size)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return RegionOfInterest.from_rectangle(frame, (x, y), (x + w, y + h))

        return RegionOfInterest(frame, mask=None)
```







```python
class FaceMeshDetector(ROIDetector):
    _lower_face = [200, 431, 411, 340, 349, 120, 111, 187, 211]

    def __init__(self, draw_landmarks=False, refine=False, **kwargs):
        super().__init__(**kwargs)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw_landmarks = draw_landmarks

    def __del__(self):
        self.face_mesh.close()

    def detect(self, frame):
        rawimg = frame.copy()

        frame.flags.writeable = False
        results = self.face_mesh.process(frame)
        frame.flags.writeable = True

        if results.multi_face_landmarks is None:
            return RegionOfInterest(frame, mask=None)

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.multi_face_landmarks,
                               tesselate=True)

        landmarks = get_facemesh_coords(results.multi_face_landmarks[0], frame)
        facerect = get_boundingbox_from_landmarks(landmarks)
        bgmask = get_default_bgmask(frame.shape[1], frame.shape[0])
        print(cv2.boundingRect(bgmask))

        return RegionOfInterest.from_contour(rawimg, landmarks[self._lower_face],
                                             facerect=facerect, bgmask=bgmask)

    def draw_facemesh(self, img, multi_face_landmarks, tesselate=False,
                      contour=False, irises=False):
        if multi_face_landmarks is None:
            return

        for face_landmarks in multi_face_landmarks:
            if tesselate:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
            if contour:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                        .get_default_face_mesh_contours_style())
            if irises and len(face_landmarks) > 468:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
```

获取facemesh检测出来的点对应笛卡尔坐标轴的坐标点：

```python
def get_facemesh_coords(landmark_list, frame):
    h, w = frame.shape[:2]
    xys = [(landmark.x, landmark.y) for landmark in landmark_list.landmark]

    return np.multiply(xys, [w, h]).astype(int)
```





### roi_detect.py



```python
class ROIDetector:
    def __init__(self, smooth_factor=0.0, **kwargs):
        self.oldroi = None
        self.smooth_factor = smooth_factor
        super().__init__(**kwargs)

    def detect(self, frame):
        raise NotImplementedError("detect method needs to be overwritten.")

    def get_roi(self, frame):
        roi = self.detect(frame)
        return roi
        # self.oldroi = exponential_smooth(roi, self.oldroi, self.smooth_factor)

        # return self.oldroi

    def __call__(self, frame):
        return self.get_roi(frame)
```



















```python
from yarppg.rppg.roi.region_of_interest import RegionOfInterest, get_default_bgmask
```



获取背景掩膜

```python
def get_default_bgmask(w, h):
    mask = np.zeros((h, w), dtype="uint8")
    cv2.rectangle(mask, (0, 0), (w, 5), 255, -1)
    # 直接255是蓝色， thickness = -1是完全填充

    return mask
```





```python
class RegionOfInterest:
    def __init__(self, base_img, mask=None, bgmask=None, facerect=None):
        self.rawimg = base_img

        self._mask = mask
        self._rectangle = None
        self._empty = True
        self._rectangular = False
        self._contours = None
        self._bgmask = bgmask
        self._facerect = facerect

        if mask is not None:
            self._rectangle = cv2.boundingRect(mask)
            self._empty = (self._rectangle[2] == 0 or self._rectangle[3] == 0)

    @classmethod
    def from_rectangle(cls, base_img, p1, p2, **kwargs):
        # https://www.pyimagesearch.com/2021/01/19/image-masking-with-opencv/
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        cv2.rectangle(mask, p1, p2, 255, cv2.FILLED)

        roi = RegionOfInterest(base_img, mask=mask, **kwargs)
        roi._rectangular = True

        return roi

    @classmethod
    def from_contour(cls, base_img, pointlist, **kwargs):
        # pointlist with shape nx2
        mask = np.zeros(base_img.shape[:2], dtype="uint8")
        contours = np.reshape(pointlist, (1, -1, 1, 2))
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)

        roi = RegionOfInterest(base_img, mask, **kwargs)
        roi._contours = contours

        return roi

    def draw_roi(self, img, color=(255, 0, 0), thickness=3):
        if self.is_empty():
            return

        if self.is_rectangular():
            p1, p2 = self.get_bounding_box(as_corners=True)
            cv2.rectangle(img, p1, p2, color, thickness)
        else:
            cv2.drawContours(img, self._contours, 0, color=color,
                             thickness=thickness)

    def pixelate_face(self, img, blursize):
        if not self.is_empty():
            xywh = self._rectangle if self._facerect is None else self._facerect
            pixelate(img, xywh, blursize)

    def is_rectangular(self):
        return self._rectangular

    def is_empty(self):
        return self._empty

    def get_bounding_box(self, as_corners=False):
        """Bounding box specified as (x, y, w, h) or min/max corners
        """
        if as_corners:
            x, y, w, h = self._rectangle
            return (x, y), (x+w, y+h)
        return self._rectangle

    def get_mean_rgb(self, background=False):
        mask = self._mask
        if background:
            if self._bgmask is None:
                raise ValueError("Background mask is not specified")
            mask = self._bgmask

        r, g, b, a = cv2.mean(self.rawimg, mask)
        return r, g, b

    def __str__(self):
        if self.is_empty():
            return "RegionOfInterest(empty)"
        if self.is_rectangular():
            return f"RegionOfInterest(rect={self._rectangle})"

        return f"RegionOfInterest(masked within bb={self._rectangle})"
```













```python
from yarppg.rppg.filters import DigitalFilter, get_butterworth_filter
```



**带通滤波器：**

```python
def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    return DigitalFilter(ba[0], ba[1])
```



**数字滤波器：**

```python
class DigitalFilter:

    def __init__(self, b, a):
        self._bs = b
        self._as = a
        self._xs = [0]*len(b)
        self._ys = [0]*(len(a)-1)

    def process(self, x):
        if np.isnan(x):  # ignore nans, and return as is
            return x

        self._xs.insert(0, x)
        self._xs.pop()
        y = (np.dot(self._bs, self._xs) / self._as[0]
             - np.dot(self._as[1:], self._ys))
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)
```







# rppg_CVPR2014

## 工具箱utils

### 0.base.utils



```python
def scale_image(image, width, height):
   	# 规划图片大小

def crop_face(image, bbx, facewidth):

   
def build_bandpass_filter(fs, order, min_freq=0.7, max_freq=4.0, plot=False):
    
   
def get_parameter(args, configuration, keyword, default):
   

```







### 1.extract_utils.py

提取ROI特征的工具包

```python
def kp66_to_mask(image, keypoints, indent=10, plot=False):

def get_mask(image, mask_points):

def get_good_features_to_track(face, npoints, quality=0.01, min_distance=10, plot=False):

def track_features(previous, current, previous_points, plot=False):

def find_transformation(previous_points, current_points):
    
def get_current_mask_points(previous_mask_points, transfo_matrix):
    
def compute_average_colors_mask(image, mask, plot=False):
 
def compute_average_colors_wholeface(image, plot=False):????为什么要计算人脸？？？
```

### 2.filter_utils.py 

滤波器的工具包

```python
import numpy


def detrend(signal, Lambda):
    """applies a detrending filter.

    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    Parameters
    ----------
    signal: numpy.ndarray
      The signal where you want to remove the trend.
    Lambda: int
      The smoothing parameter.

    Returns
    -------
    filtered_signal: numpy.ndarray
      The detrended signal.

    """
    signal_length = signal.shape[0]

    # observation matrix
    H = numpy.identity(signal_length)

    # second-order difference matrix
    from scipy.sparse import spdiags
    ones = numpy.ones(signal_length)
    minus_twos = -2 * numpy.ones(signal_length)
    diags_data = numpy.array([ones, minus_twos, ones])
    diags_index = numpy.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = numpy.dot((H - numpy.linalg.inv(H + (Lambda ** 2) * numpy.dot(D.T, D))), signal)
    return filtered_signal


def average(signal, window_size):
    """Moving average filter.

    Parameters
    ----------
    signal: numpy.ndarray
      The signal to filter.
    window_size: int
      The size of the window to compute the average.

    Returns
    -------
    filtered_signal: numpy.ndarray
      The averaged signal.

    """
    from scipy.signal import lfilter
    a = 1.0
    b = numpy.zeros(window_size)
    b += (1.0 / float(window_size))
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal
```

### 3.illum_utils.py

去光照工具箱

```python
def rectify_illumination(face_color, bg_color, step, length):


def nlms(signal, desired_signal, n_filter_taps, step, initCoeffs=None, adapt=True):
```

### 4.motion_utils.py

```python
def build_segments(signal, length):
	# 信号被分成规定长度的段（无重叠），不同的段被堆叠。


def prune_segments(segments, threshold):
    # 把超过阈值的部分切割
    
def build_final_signal(segments, gaps):
 	# builds the final signal with remaining segments.   
```

















## 1.extract_face_and_bg_signals.py

导入的包：

```python
from ...base.utils import crop_face

from ..extract_utils import kp66_to_mask
from ..extract_utils import get_good_features_to_track
from ..extract_utils import track_features
from ..extract_utils import find_transformation
from ..extract_utils import get_current_mask_points
from ..extract_utils import get_mask
from ..extract_utils import compute_average_colors_mask
from ..extract_utils import compute_average_colors_wholeface
```





```
bounding_boxes = obj.load_face_detection()

prev_face = crop_face(frame, prev_bb, facewidth)
prev_features = get_good_features_to_track(face, npoints, quality, distance, plot)

face_mask = get_mask(frame, mask_points)

face_color[i] = compute_average_colors_mask(frame, face_mask, plot)[1]


bg_mask = numpy.zeros((frame.shape[1], frame.shape[2]), dtype=bool)
bg_mask[:100, :100] = True
bg_color[i] = compute_average_colors_mask(frame, bg_mask, plot)[1]
```

存的就是face_color和bg_color





```python
# load video
video = obj.load_video(configuration.dbdir)
logger.info("Processing input video from `%s'...", video.filename)

# load the result of face detection
bounding_boxes = obj.load_face_detection()

# average green color in the mask area
face_color = numpy.zeros(len(video), dtype='float64')
# average green color in the background area
bg_color = numpy.zeros(len(video), dtype='float64')

# loop on video frames
for i, frame in enumerate(video):
    logger.debug("Processing frame %d/%d...", i + 1, len(video))

    if i == 0:
        # first frame:
        # -> load the keypoints detected by DMRF
        # -> infer the mask from the keypoints
        # -> detect the face
        # -> get "good features" inside the face
        if not wholeface:
            kpts = obj.load_drmf_keypoints()
            mask_points, mask = kp66_to_mask(frame, kpts, indent, plot)

        try:
            bbox = bounding_boxes[i]
        except NameError:
            bbox, quality = bob.ip.facedetect.detect_single_face(frame)

        # define the face width for the whole sequence
        facewidth = bbox.size[1]
        face = crop_face(frame, bbox, facewidth)

        if not wholeface:
            good_features = get_good_features_to_track(face, npoints, quality, distance, plot)
    else:
        # subsequent frames:
        # -> crop the face with the bounding_boxes of the previous frame (so
        #    that faces are of the same size)
        # -> get the projection of the corners detected in the previous frame
        # -> find the (affine) transformation relating previous corners with
        #    current corners
        # -> apply this transformation to the mask
        face = crop_face(frame, prev_bb, facewidth)
        if not wholeface:
            good_features = track_features(prev_face, face, prev_features, plot)
            project = find_transformation(prev_features, good_features)
            if project is None:
                logger.warn("Sequence {0}, frame {1} : No projection was found"
                            " between previous and current frame, mask from previous frame will be used"
                            .format(obj.path, i))
            else:
                mask_points = get_current_mask_points(mask_points, project)

    # update stuff for the next frame:
    # -> the previous face is the face in this frame, with its bbox (and not
    #    with the previous one)
    # -> the features to be tracked on the next frame are re-detected
    try:
        prev_bb = bounding_boxes[i]
    except NameError:
        bb, quality = bob.ip.facedetect.detect_single_face(frame)
        prev_bb = bb

    if not wholeface:
        prev_face = crop_face(frame, prev_bb, facewidth)
        prev_features = get_good_features_to_track(face, npoints, quality, distance, plot)
        if prev_features is None:
            logger.warn("Sequence {0}, frame {1} No features to track"
                        " detected in the current frame, using the previous ones"
                        .format(obj.path, i))
            prev_features = good_features

        # get the bottom face region average colors
        face_mask = get_mask(frame, mask_points)
        # original algorithm: green only
        face_color[i] = compute_average_colors_mask(frame, face_mask, plot)[1]
    else:
        face_color[i] = compute_average_colors_wholeface(face, plot)

    # get the background region average colors
    bg_mask = numpy.zeros((frame.shape[1], frame.shape[2]), dtype=bool)
    bg_mask[:100, :100] = True
    bg_color[i] = compute_average_colors_mask(frame, bg_mask, plot)[1]
```











## 2.illumination_rectification.py



```python
start = get_parameter(args, configuration, 'start', 0)
end = get_parameter(args, configuration, 'end', 0)

start_index = start
end_index = end

if (end_index == 0):
     end_index = face.shape[0]
if end_index > face.shape[0]:
     logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, face.shape[0]))

# truncate(截短) the signals if needed
face = face[start_index:end_index]
bg = bg[start_index:end_index]


# apply NLMS filtering
corrected_green = rectify_illumination(face, bg, step, length)  # step == 0.05  lenth == 1


        if plot:
            from matplotlib import pyplot
            f, axarr = pyplot.subplots(3, sharex=True)
            axarr[0].plot(range(face.shape[0]), face, 'g')
            axarr[0].set_title(r"$g_{face}$: average green value on the mask region")
            axarr[1].plot(range(bg.shape[0]), bg, 'g')
            axarr[1].set_title(r"$g_{bg}$: average green value on the background")
            axarr[2].plot(range(corrected_green.shape[0]), corrected_green, 'g')
            axarr[2].set_title(r"$g_{IR}$: illumination rectified signal")
            pyplot.show()
```

输出：



<img src="E:\Picture_bin\2022_5\111dwd.png" style="zoom:150%;" />







```python
 # load the color signal of the background
        bg_file = obj.make_path(bgdir, '.hdf5')
        try:
            bg = bob.io.base.load(bg_file)
        except (IOError, RuntimeError) as e:
            logger.warn("Skipping file `%s' (no background file available)", obj.path)
            continue

        # indices where to start and to end the processing
        logger.debug("Sequence length = {0}".format(face.shape[0]))
        start_index = start
        end_index = end
        if (end_index == 0):
            end_index = face.shape[0]
        if end_index > face.shape[0]:
            logger.warn("Skipping Sequence {0} : not long enough ({1})".format(obj.path, face.shape[0]))
            continue

        logger.info("Processing sequence {0} ...".format(obj.path))

        # truncate the signals if needed
        face = face[start_index:end_index]
        bg = bg[start_index:end_index]
        logger.debug("Processing %d frames...", face.shape[0])

        # apply NLMS filtering
        corrected_green = rectify_illumination(face, bg, step, length)      # step == 0.05  lenth == 1

```



```python
f, axarr = pyplot.subplots(3, sharex=True)
axarr[0].plot(range(face.shape[0]), face, 'g')
axarr[0].set_title(r"$g_{face}$: average green value on the mask region")
axarr[1].plot(range(bg.shape[0]), bg, 'g')
axarr[1].set_title(r"$g_{bg}$: average green value on the background")
axarr[2].plot(range(corrected_green.shape[0]), corrected_green, 'g')
axarr[2].set_title(r"$g_{IR}$: illumination rectified signal")
pyplot.show()
```

## 3.motion_elimination.py



```python
  -i, --illumdir=<path>         The path to the saved illumination corrected signal
                                on your disk [default: illumination].
  -o, --motiondir=<path>        The path to the output directory where the resulting
                                motion corrected signals will be stored
                                [default: motion].
  -L, --seglength=<int>         The length of the segments [default: 61]
      --cutoff=<float>          Specify the percentage of largest segments to
                                determine the threshold [default: 0.05].
   	  --save-threshold=<path>   Save the found threshold to cut segments [default: 										threshold.txt]. 
      --load-threshold=<path>   Load the threshold to cut segments [default: None]. 




from ...base.utils import get_parameter
from ..motion_utils import build_segments
from ..motion_utils import prune_segments
from ..motion_utils import build_final_signal
from ..motion_utils import build_final_signal_cvpr14
```





```python
illum_file = obj.make_path(illumdir, '.hdf5')


# get the standard deviation in the segments
green_segments, __ = build_segments(color, seglength)
std_green = numpy.std(green_segments, 1, ddof=1)
all_stds.extend(std_green.tolist())

# sort the std and find the 5% at the top to get the threshold
sorted_stds = sorted(all_stds, reverse=True)
cut_index = int(cutoff * len(all_stds)) + 1
threshold = sorted_stds[cut_index]




```







## 4.filter.py

```python
  -i, --motiondir=<path>     The path to the saved signals to be filtered on
                            your disk [default: motion].
  -o, --pulsedir=<path>     The path to the output directory where the resulting
                            color signals will be stored [default: filtered].
  --Lambda=<int>            Lambda parameter for detrending (see article) [default: 300]
  --window=<int>            Moving window length [default: 23]
  -f, --framerate=<int>     Frame-rate of the video sequence [default: 61]
  --order=<int>             Bandpass filter order [default: 128]    阶数
```







```python
from ..filter_utils import detrend
from ..filter_utils import average
from ...base.utils import build_bandpass_filter
from ...base.utils import get_parameter
```



```python
from ..filter_utils import detrend
```



```python
def detrend(signal, Lambda):
    """applies a detrending filter.

    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    Parameters
    ----------
    signal: numpy.ndarray
      The signal where you want to remove the trend.
    Lambda: int
      The smoothing parameter.

    Returns
    -------
    filtered_signal: numpy.ndarray
      The detrended signal.

    """
    signal_length = signal.shape[0]

    # observation matrix
    H = numpy.identity(signal_length)

    # second-order difference matrix
    from scipy.sparse import spdiags
    ones = numpy.ones(signal_length)
    minus_twos = -2 * numpy.ones(signal_length)
    diags_data = numpy.array([ones, minus_twos, ones])
    diags_index = numpy.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = numpy.dot((H - numpy.linalg.inv(H + (Lambda ** 2) * numpy.dot(D.T, D))), signal)
    return filtered_signal
```





```python
from ..filter_utils import average
```

滑动均值滤波器

```python
def average(signal, window_size):
    """
    Moving average filter.

    Parameters
    ----------
    signal: numpy.ndarray
      The signal to filter.
    window_size: int
      The size of the window to compute the average.

    Returns
    -------
    filtered_signal: numpy.ndarray
      The averaged signal.

    """
    from scipy.signal import lfilter
    a = 1.0
    b = numpy.zeros(window_size)
    b += (1.0 / float(window_size))
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal
```





```python
from ...base.utils import build_bandpass_filter
```



```python
b = build_bandpass_filter(framerate, order, plot)
```

```
'framerate', 61
'order', 128	阶数
'plot' ：False  不用画出滤波器
```





```python
def build_bandpass_filter(fs, order, min_freq=0.7, max_freq=4.0, plot=False):
    """builds a butterworth bandpass filter.

    Parameters
    ----------
    fs: float
      sampling frequency of the signal (i.e. framerate).
    order: int
      The order of the filter (the higher, the sharper).
    min_freq: int
      The order of the filter (the higher, the sharper).
    order: int
      The order of the filter (the higher, the sharper).
    plot: bool
      Plots the frequency response of the filter.

    Returns
    -------
    b: numpy.ndarray
      The coefficients of the FIR filter.

    """
    # frequency range in Hertz, corresponds to plausible heart-rate values, i.e. [42-240] beats per minute
    min_freq = 0.7
    max_freq = 4.0

    from scipy.signal import firwin
    nyq = fs / 2.0
    numtaps = order + 1
    b = firwin(numtaps, [min_freq / nyq, max_freq / nyq], pass_zero=False)

    # show the frequency response of the filter
    if plot:
        from matplotlib import pyplot
        from scipy.signal import freqz
        w, h = freqz(b)
        fig = pyplot.figure()
        pyplot.title('Bandpass filter frequency response')
        pyplot.plot(w * fs / (2 * numpy.pi), 20 * numpy.log10(abs(h)), 'b')
        pyplot.axvline(x=min_freq, color="red")
        pyplot.axvline(x=max_freq, color="red")
        pyplot.ylabel('Amplitude [dB]', color='b')
        pyplot.xlabel('Frequency [Hz]')
        pyplot.show()

    return b
```



```python
from ...base.utils import get_parameter
```



```python
def get_parameter(args, configuration, keyword, default):
    """ Get the right value for a parameter

    The parameters are either defined in a separate configuration file
    or given directly via command-line. Note that the command-line
    has priority over the configuration file.

    As a convention, parameters made with more than one word
    are provided with an underscore in the configuration file, and with an
    hyphen in the command-line:

      - configuration:  batch_size=64
      - command line:   --batch-size=64

    Parameters
    ----------
    args: dictionary
      The arguments as parsed from the command line.
    configuration: object
      The arguments given by the configuration file.
    keyword: string
      the keyword for the parameter to process (in the "configuration" style)
    default:
      The default value of the parameter

    Returns
    -------
    arg:
      The right value for the given keyword argument

    """

    # get the keyword in a "docopt" friendly format
    args_kw = '--' + keyword.replace('_', '-')

    # get the type of this argument
    _type = type(default)

    # get the default value
    arg_default = default

    # get the argument in the configuration file
    if hasattr(configuration, keyword):
        arg_config = getattr(configuration, keyword)
    else:
        arg_config = None

    # get the argument from the command-line
    if default is not None:
        arg_command = _type(args[args_kw])
    else:
        arg_command = default

        # if the argument was not specified in the config file
    if not hasattr(configuration, keyword):
        return arg_command
    else:
        if (arg_command == arg_default):
            return arg_config
        else:
            return arg_command
```







```python
def main(user_input=None):
    # Parse the command-line arguments
    if user_input is not None:
        arguments = user_input
    else:
        arguments = sys.argv[1:]

    prog = os.path.basename(sys.argv[0])
    completions = dict(prog=prog, version=version, )
    args = docopt(__doc__ % completions, argv=arguments, version='Filtering for signals (%s)' % version, )

    # load configuration file
    configuration = load([os.path.join(args['<configuration>'])])

    # get various parameters, either from config file or command-line
    protocol = get_parameter(args, configuration, 'protocol', 'all')
    subset = get_parameter(args, configuration, 'subset', None)
    motiondir = get_parameter(args, configuration, 'motiondir', 'motion')
    pulsedir = get_parameter(args, configuration, 'pulsedir', 'pulse')
    Lambda = get_parameter(args, configuration, 'Lambda', 300)
    window = get_parameter(args, configuration, 'window', 23)
    framerate = get_parameter(args, configuration, 'framerate', 61)
    order = get_parameter(args, configuration, 'order', 128)
    overwrite = get_parameter(args, configuration, 'overwrite', False)
    plot = get_parameter(args, configuration, 'plot', False)
    gridcount = get_parameter(args, configuration, 'gridcount', False)
    verbosity_level = get_parameter(args, configuration, 'verbose', 0)

    # if the user wants more verbosity, lowers the logging level
    from bob.core.log import set_verbosity_level
    set_verbosity_level(logger, verbosity_level)

    # TODO: find a way to check protocol names - Guillaume HEUSCH, 22-06-2018
    if hasattr(configuration, 'database'):
        objects = configuration.database.objects(protocol, subset)
    else:
        logger.error("Please provide a database in your configuration file !")
        sys.exit()

    # if we are on a grid environment, just find what I have to process.
    sge = False
    try:
        sge = os.environ.has_key('SGE_TASK_ID')  # python2
    except AttributeError:
        sge = 'SGE_TASK_ID' in os.environ  # python3

    if sge:
        pos = int(os.environ['SGE_TASK_ID']) - 1
        if pos >= len(objects):
            raise RuntimeError("Grid request for job {} on a setup with {} jobs".format(pos, len(objects)))
        objects = [objects[pos]]

    if gridcount:
        print(len(objects))
        sys.exit()

    # build the bandpass filter one and for all
    b = build_bandpass_filter(framerate, order, plot)

    ################
    ### LET'S GO ###
    ################
    for obj in objects:

        # expected output file
        output = obj.make_path(pulsedir, '.hdf5')

        # if output exists and not overwriting, skip this file
        if os.path.exists(output) and not overwrite:
            logger.info("Skipping output file `%s': already exists, use --overwrite to force an overwrite", output)
            continue

        # load the corrected color signals of shape (3, nb_frames)
        logger.info("Filtering in signal from `%s'...", obj.path)
        motion_file = obj.make_path(motiondir, '.hdf5')
        try:
            motion_corrected_signal = bob.io.base.load(motion_file)
        except (IOError, RuntimeError) as e:
            logger.warn("Skipping file `%s' (no motion corrected signal file available)", obj.path)
            continue

        # check whether the signal is long enough to be filtered with the bandpass of this order
        padlen = 3 * len(b)
        if motion_corrected_signal.shape[0] < padlen:
            logger.warn(
                "Skipping file {0} (unable to bandpass filter it, the signal is probably not long enough)".format(
                    obj.path))
            continue

        # detrend
        green_detrend = detrend(motion_corrected_signal, Lambda)
        # average
        green_averaged = average(green_detrend, window)
        # bandpass
        from scipy.signal import filtfilt
        green_bandpassed = filtfilt(b, numpy.array([1]), green_averaged)
		# 去趋势、均值平滑滤波最后带通
        
        # plot the result
        if plot:
            from matplotlib import pyplot
            f, ax = pyplot.subplots(4, sharex=True)
            ax[0].plot(range(motion_corrected_signal.shape[0]), motion_corrected_signal, 'g')
            ax[0].set_title('Original signal')
            ax[1].plot(range(motion_corrected_signal.shape[0]), green_detrend, 'g')
            ax[1].set_title('After detrending')
            ax[2].plot(range(motion_corrected_signal.shape[0]), green_averaged, 'g')
            ax[2].set_title('After averaging')
            ax[3].plot(range(motion_corrected_signal.shape[0]), green_bandpassed, 'g')
            ax[3].set_title('Bandpassed signal')
            pyplot.show()

        output_data = numpy.copy(green_bandpassed)

        # saves the data into an HDF5 file with a '.hdf5' extension
        pulse_outdir = os.path.dirname(output)
        if not os.path.exists(pulse_outdir): bob.io.base.create_directories_safe(pulse_outdir)
        bob.io.base.save(output_data, output)
        logger.info("Output file saved to `%s'...", output)

    return 0
```



## 5.video2skin.py

```
Skin color extraction for database videos
```















