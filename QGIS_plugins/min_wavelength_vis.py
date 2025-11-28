from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel
from qgis.gui import QgsMapTool
from qgis.core import QgsMapLayer, QgsRaster
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

# ----- Built-in quickhull2d implementation -----
# ----- The following code is adapted from https://github.com/wimhbakker/hyppy -----
link = lambda a,b: np.concatenate((a,b[1:]))
edge = lambda a,b: np.concatenate(([a],[b]))

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
    """Resample hull to match original wavelengths"""
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
    """Compute and resample upper convex hull"""
    return resample(qhulltop(sample), sample)

def proper_continuum_curve(wavs, spec):
    """Extract continuum using convex hull"""
    points = np.column_stack([wavs, spec])
    curve = hull_resampled(points)
    continuum = np.interp(wavs, curve[:,0], curve[:,1])
    return continuum, curve[:,0], curve[:,1]

def hull_removed_spectrum(wavs, spec):
    """Remove continuum from spectrum by division"""
    continuum, hull_wavs, hull_spec = proper_continuum_curve(wavs, spec)
    return continuum, hull_wavs, hull_spec, spec / continuum

def fit_parabola_and_curve(x, y):
    """Fit parabola through 3 points"""
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

def get_wavelengths_from_raster(raster_layer):
    """
    Extract wavelengths from raster metadata.
    Returns wavelengths in nm, or None if not found.
    """
    provider = raster_layer.dataProvider()
    n_bands = raster_layer.bandCount()
    
    # Check GDAL metadata domains
    metadata_domains = ['', 'ENVI']  # Common domains
    
    for domain in metadata_domains:
        try:
            metadata = provider.metadata(domain)
            
            # Look for wavelength information
            for key, value in metadata.items():
                if 'wavelength' in key.lower():
                    try:
                        # Try parsing as comma-separated
                        if ',' in value:
                            wavelengths = [float(w.strip()) for w in value.split(',')]
                            if len(wavelengths) == n_bands:
                                return np.array(wavelengths)
                        # Try parsing as array string
                        elif '[' in value or '{' in value:
                            import re
                            numbers = re.findall(r'[-+]?\d*\.?\d+', value)
                            wavelengths = [float(n) for n in numbers]
                            if len(wavelengths) == n_bands:
                                return np.array(wavelengths)
                    except:
                        continue
        except:
            continue
    
    return None

def get_sensor_info(raster_layer):
    """
    Determine sensor type and wavelengths.
    Returns: (sensor_name, wavelengths_nm)
    """
    n_bands = raster_layer.bandCount()
    
    # Try to extract from metadata first
    wavelengths = get_wavelengths_from_raster(raster_layer)
    
    if wavelengths is not None:
        # Check units (convert to nm if needed)
        if np.max(wavelengths) < 100:  # Likely micrometers
            wavelengths = wavelengths * 1000  # Convert to nm
        
        # Determine sensor by characteristics
        # !! adjust here as needed for known sensors
        if n_bands == 9 and 500 < np.min(wavelengths) < 3000:
            return "ASTER", wavelengths
        elif 250 <= n_bands <= 300 and 300 < np.min(wavelengths) < 3000:
            return "EMIT", wavelengths
        else:
            return f"Unknown sensor ({n_bands} bands)", wavelengths
    
    # Fallback: Known sensor patterns
    if n_bands == 9:
        return "ASTER (assumed)", np.array([560, 660, 820, 1650, 2165, 2205, 2260, 2330, 2400])
    elif n_bands == 285:
        return "EMIT (estimated)", np.linspace(380, 2500, 285)
    else:
        # Generic fallback: use band numbers
        iface.messageBar().pushMessage(
            "Warning", 
            f"Wavelengths not found. Using band numbers for {n_bands} bands.",
            level=1, duration=5
        )
        return f"Unknown ({n_bands} bands)", np.arange(1, n_bands + 1)

def get_pixel_values(raster_layer, qgs_point):
    """Extract all band values at clicked location"""
    provider = raster_layer.dataProvider()
    ident = provider.identify(qgs_point, QgsRaster.IdentifyFormatValue)
    if not ident.isValid():
        return []
    band_results = ident.results()
    n_bands = raster_layer.bandCount()
    return [band_results.get(b, None) for b in range(1, n_bands+1)]

class FlexibleMinWavDialog(QDialog):
    """Enhanced dialog that works with any hyperspectral sensor"""
    
    def __init__(self, sensor_name, wavs, spec, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Minimum Wavelength Visualisation - {sensor_name}")
        self.setWindowFlags(Qt.Window)
        
        outer_layout = QVBoxLayout(self)
        
        # Add info label
        info_label = QLabel(f"Sensor: {sensor_name} | Bands: {len(spec)} | Wavelength range: {wavs.min():.1f} - {wavs.max():.1f} nm")
        outer_layout.addWidget(info_label)
        
        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.figure)
        outer_layout.addWidget(self.canvas)
        
        self.sensor_name = sensor_name
        self.wavs = wavs
        self.spec = spec
        self._do_plots()
    
    def _do_plots(self):
        """Create 6-panel visualization"""
        bands = np.arange(1, len(self.spec)+1)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(self.spec)
        if not np.any(valid_mask):
            iface.messageBar().pushMessage("Error", "All values are NaN", level=2, duration=3)
            return
        
        wavs_valid = self.wavs[valid_mask]
        spec_valid = self.spec[valid_mask]
        
        # Continuum removal
        continuum, hull_wavs, hull_spec, hull_removed = hull_removed_spectrum(wavs_valid, spec_valid)
        
        # Find minimum
        min_idx = np.argmin(hull_removed)
        min_wav = wavs_valid[min_idx]
        min_val = hull_removed[min_idx]
        
        # Parabola fit
        show_parabola = min_idx > 0 and min_idx < len(wavs_valid)-1
        xfit = yfit = x_curve = y_curve = zx = zy = None
        
        if show_parabola:
            xfit = np.array([wavs_valid[min_idx-1], wavs_valid[min_idx], wavs_valid[min_idx+1]])
            yfit = np.array([hull_removed[min_idx-1], hull_removed[min_idx], hull_removed[min_idx+1]])
            params, zx, zy, x_curve, y_curve, coeffs = fit_parabola_and_curve(xfit, yfit)
        
        # Create plots
        self.figure.clf()
        axs = self.figure.subplots(2, 3)
        
        # 1. Band number vs band value
        ax = axs[0,0]
        ax.plot(bands, self.spec, 'ko-', label='Band Value', markersize=3)
        ax.set_xlabel("Band Number")
        ax.set_ylabel("Pixel Value")
        ax.set_title("1: Band vs Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Wavelength vs band value
        ax = axs[0,1]
        ax.plot(self.wavs, self.spec, 'ko-', label='Pixel Value', markersize=3)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Pixel Value")
        ax.set_title("2: Wavelength vs Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Wavelength, band value & convex hull/continuum
        ax = axs[0,2]
        ax.plot(wavs_valid, spec_valid, 'ko-', label='Pixel Value', markersize=3)
        ax.plot(hull_wavs, hull_spec, 'g^--', label='Hull Points', markersize=5)
        ax.plot(wavs_valid, continuum, 'g-', lw=2, label='Continuum')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Pixel Value")
        ax.set_title("3: Continuum Overlay")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4: Wavelength vs continuum-removed value
        ax = axs[1,0]
        ax.plot(wavs_valid, hull_removed, 'bo-', label='Continuum-Removed', markersize=3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("4: Continuum-Removed")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5: Highlight global minimum
        ax = axs[1,1]
        ax.plot(wavs_valid, hull_removed, 'bo-', label='Continuum-Removed', markersize=3)
        ax.plot(min_wav, min_val, 'ro', ms=10, label=f"Min: {min_wav:.1f} nm")
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("5: Global Minimum")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6: Fit parabola at minimum
        ax = axs[1,2]
        ax.plot(wavs_valid, hull_removed, 'bo-', label='Continuum-Removed', markersize=3)
        ax.plot(min_wav, min_val, 'ro', ms=10, label=f"Min: {min_wav:.1f} nm")
        
        if show_parabola and x_curve is not None:
            ax.plot(xfit, yfit, 'gs', ms=8, label='Neighbors')
            ax.plot(x_curve, y_curve, 'm-', lw=2, label='Parabola')
            ax.plot(zx, zy, 'md', ms=11, label=f'Vertex: {zx:.1f} nm')
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Value (Hull-removed)")
        ax.set_title("6: Parabola Fit")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

class RasterClickTool(QgsMapTool):
    """Map tool for clicking raster pixels"""
    
    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback
    
    def canvasReleaseEvent(self, event):
        point = self.canvas.getCoordinateTransform().toMapCoordinates(
            event.pos().x(), event.pos().y()
        )
        layer = self.canvas.currentLayer()
        
        if layer and layer.type() == QgsMapLayer.RasterLayer:
            self.callback(layer, point)
        else:
            iface.messageBar().pushMessage(
                "Error", 
                "Please select a raster layer.", 
                level=2, duration=2
            )

def on_pixel_clicked(layer, point):
    """Callback when user clicks a pixel"""
    # Get sensor info and wavelengths
    sensor_name, wavelengths = get_sensor_info(layer)
    
    # Get pixel values
    vals = get_pixel_values(layer, point)
    vals = np.array(vals, dtype=float)
    
    if len(vals) != len(wavelengths):
        iface.messageBar().pushMessage(
            "Error", 
            f"Band count mismatch: {len(vals)} values vs {len(wavelengths)} wavelengths",
            level=2, duration=3
        )
        return
    
    # Create and show dialog
    dialog = FlexibleMinWavDialog(sensor_name, wavelengths, vals, parent=iface.mainWindow())
    dialog.show()

# Activate the tool
canvas = iface.mapCanvas()
click_tool = RasterClickTool(canvas, on_pixel_clicked)
canvas.setMapTool(click_tool)

iface.messageBar().pushMessage(
    "Ready", 
    "Click any pixel in the raster to visualise spectrum and continuum removal",
    level=0, duration=5
)