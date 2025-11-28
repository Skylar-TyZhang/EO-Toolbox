from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout
from qgis.gui import QgsMapTool
from qgis.core import QgsMapLayer, QgsRaster
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

# ---- ASTER band central wavelengths (nm) ----
aster_wavelengths_nm = np.array([560, 660, 820, 1650, 2165, 2205, 2260, 2330, 2400])

# ----- Built-in quickhull2d implementation -----
link = lambda a,b: np.concatenate((a,b[1:]))
edge = lambda a,b: np.concatenate(([a],[b]))
def qhull(sample):
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, axis=0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base
    if len(sample) > 2:
        axis = sample[:,0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return link(dome(sample, base),
                    dome(sample, base[::-1]))
    else:
        return sample
def qhulltop(sample):
    def dome(sample,base):
        h, t = base
        dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
        outer = np.repeat(sample, dists>0, axis=0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base
    if len(sample) > 2:
        axis = sample[:,0]
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], axis=0)
        return dome(sample, base)
    else:
        return sample
def resample(hull, sample):
    xs = sample[:, 0].copy()
    xs.sort()
    ys = np.zeros(xs.shape)
    xhull = hull[:, 0]
    yhull = hull[:, 1]
    for i in range(len(xs)):
        i_hull = xhull.searchsorted(xs[i])
        if i_hull == 0:
            ys[i] = yhull[0]
        elif i_hull == len(hull):
            ys[i] = yhull[-1]
        elif xs[i] == xhull[i_hull]:
            ys[i] = yhull[i_hull]
        else:
            i_left = i_hull - 1
            i_right = i_hull
            ys[i] = (xs[i] - xhull[i_left]) / (xhull[i_right] - xhull[i_left]) * (yhull[i_right] - yhull[i_left]) + yhull[i_left]
    return np.vstack((xs, ys)).T
def hull_resampled(sample):
    return resample(qhulltop(sample), sample)

def proper_continuum_curve(wavs, spec):
    points = np.column_stack([wavs, spec])
    curve = hull_resampled(points)
    continuum = np.interp(wavs, curve[:,0], curve[:,1])
    return continuum, curve[:,0], curve[:,1]

def hull_removed_spectrum(wavs, spec):
    continuum, hull_wavs, hull_spec = proper_continuum_curve(wavs, spec)
    return continuum, hull_wavs, hull_spec, spec / continuum

def fit_parabola_and_curve(x, y):
    from scipy.optimize import leastsq
    def model(params, x): return params[0]*x*x + params[1]*x + params[2]
    def residuals(p, y, x): return y - model(p, x)
    if len(x) != 3 or len(y) != 3:
        return None, None, None, None, None, None
    params = leastsq(residuals, [1,1,1], args=(y, x))[0]
    a, b, c = params
    zx = -b/(2*a) if a != 0 else x[1]
    zy = model(params, zx)
    x_range = np.linspace(min(x)-50, max(x)+50, 100)
    y_curve = model(params, x_range)
    return params, zx, zy, x_range, y_curve, (a, b, c)

def get_pixel_values(raster_layer, qgs_point):
    provider = raster_layer.dataProvider()
    ident = provider.identify(qgs_point, QgsRaster.IdentifyFormatValue)
    if not ident.isValid():
        return []
    band_results = ident.results()
    n_bands = raster_layer.bandCount()
    return [band_results.get(b, None) for b in range(1, n_bands+1)]

class SixTileMinWavDialog(QDialog):
    def __init__(self, wavs, spec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Minimum Wavelength Process Visualization (ASTER)")
        self.setWindowFlags(Qt.Window)  # <-- This enables minimize, maximize, close
        outer_layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.figure)
        outer_layout.addWidget(self.canvas)
        self.wavs = wavs
        self.spec = spec
        self._do_plots()
    def _do_plots(self):
        bands = np.arange(1, len(self.spec)+1)
        continuum, hull_wavs, hull_spec, hull_removed = hull_removed_spectrum(self.wavs, self.spec)
        min_idx = np.argmin(hull_removed)
        min_wav = self.wavs[min_idx]
        min_val = hull_removed[min_idx]
        show_parabola = min_idx > 0 and min_idx < len(self.wavs)-1
        # Parabola fit
        xfit = yfit = params = zx = zy = x_curve = y_curve = None
        if show_parabola:
            xfit = np.array([self.wavs[min_idx-1], self.wavs[min_idx], self.wavs[min_idx+1]])
            yfit = np.array([hull_removed[min_idx-1], hull_removed[min_idx], hull_removed[min_idx+1]])
            params, zx, zy, x_curve, y_curve, coeffs = fit_parabola_and_curve(xfit, yfit)
        self.figure.clf()
        axs = self.figure.subplots(2, 3)
        # 1. Band number vs band value
        ax = axs[0,0]
        ax.plot(bands, self.spec, 'ko-', label='Band Value')
        ax.set_xticks(bands)
        ax.set_xlabel("Band Number")
        ax.set_ylabel("Pixel Value")
        ax.set_title("1: Band vs Value")
        ax.grid(True)
        ax.legend()
        # 2. Wavelength vs band value
        ax = axs[0,1]
        ax.plot(self.wavs, self.spec, 'ko-', label='Pixel Value')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Pixel Value")
        ax.set_title("2: Wavelength vs Value")
        ax.grid(True)
        ax.legend()
        # 3. Wavelength, band value & convex hull/continuum
        ax = axs[0,2]
        ax.plot(self.wavs, self.spec, 'ko-', label='Pixel Value')
        ax.plot(hull_wavs, hull_spec, 'g^--', label='Convex Hull Pts')
        ax.plot(self.wavs, continuum, 'g-', lw=2, label='Continuum (Convex Hull)')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Pixel Value")
        ax.set_title("3: Add Proper Continuum")
        ax.grid(True)
        ax.legend()
        # 4: Wavelength vs continuum-removed value
        ax = axs[1,0]
        ax.plot(self.wavs, hull_removed, 'bo-', label='Continuum-Removed')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("4: Continuum-Removed Spectrum")
        ax.grid(True)
        ax.legend()
        # 5: Highlight global minimum
        ax = axs[1,1]
        ax.plot(self.wavs, hull_removed, 'bo-', label='Continuum-Removed')
        ax.plot(min_wav, min_val, 'ro', ms=10, label=f"Global Min ({min_wav:.0f} nm)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("5: Show Global Minimum")
        ax.grid(True)
        ax.legend()
        # 6: Fit parabola at minimum (curve included)
        ax = axs[1,2]
        ax.plot(self.wavs, hull_removed, 'bo-', label='Continuum-Removed')
        ax.plot(min_wav, min_val, 'ro', ms=10, label=f"Global Min ({min_wav:.0f} nm)")
        if show_parabola:
            ax.plot(xfit, yfit, 'gs', ms=8, label='Parabola Neighbors')
            ax.plot(x_curve, y_curve, 'm-', lw=2, label='Parabola Fit')
            ax.plot(zx, zy, 'md', ms=11, label='Parabola Vertex')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("6: Parabola Fit to Minimum")
        ax.grid(True)
        ax.legend()
        self.canvas.draw()

class RasterClickTool(QgsMapTool):
    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback
    def canvasReleaseEvent(self, event):
        point = self.canvas.getCoordinateTransform().toMapCoordinates(event.pos().x(), event.pos().y())
        layer = self.canvas.currentLayer()
        if layer and layer.type() == QgsMapLayer.RasterLayer:
            self.callback(layer, point)
        else:
            iface.messageBar().pushMessage("Select a raster layer.", level=2, duration=2)
def on_pixel_clicked(layer, point):
    vals = get_pixel_values(layer, point)
    vals = np.array(vals, dtype=float)
    # if len(vals) != 9:
    #     iface.messageBar().pushMessage("Only works for ASTER 9 bands!", level=2, duration=3)
    #     return
    dialog = SixTileMinWavDialog(aster_wavelengths_nm, vals, parent=iface.mainWindow())
    dialog.show()
canvas = iface.mapCanvas()
click_tool = RasterClickTool(canvas, on_pixel_clicked)
canvas.setMapTool(click_tool)
iface.messageBar().pushMessage("MinWavelength: Click on ASTER pixel for convex hull stepwise plot (with parabola)", duration=5)