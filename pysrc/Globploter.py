import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime, timezone
import ffmpeg
import TLE


class GlobPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.ax.add_feature(cfeature.LAND, facecolor='#e6d5b8', edgecolor='none')
        self.ax.add_feature(cfeature.OCEAN, facecolor='#a3d8ff')
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#666666', alpha=0.7)
        self.ax.set_global()
        self.title = self.ax.set_title("World Map - Animated Moving Points", fontsize=16, pad=20)
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False   
        gl.right_labels = False
        self.ani = None  # Will hold the animation object

        self.satellite_data = []  # List to hold satellite info for animation
        self.scatters = [] 
        self.texts = []
        self.trails = []
        # Animation settings
        self.interval_ms = 3000          # Speed of animation (lower = faster)
        self.repeat = True              # Whether to loop the animation
        self.is_paused = False
        self.current_frame = 0

        self.times = None  # Will be set when adding satellites

        # Marker settings
        self.marker_size = 50
        self.marker_edge_color = 'white'
        self.marker_edge_width = 0.5

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def get_facecolor_by_inclination(self, inclination):
        if inclination < 50:
            return 'green'
        elif inclination >= 50 and inclination < 60:
            return 'yellow'
        elif inclination >= 60 and inclination < 75:
            return 'orange'
        else:
            return 'red'

    def add_moving_point(self, satellite:TLE):
        self.satellite_data.append(satellite)
        if self.times is None:
            self.times = satellite.get_times()
            
        color = self.get_facecolor_by_inclination(satellite.get_inclination())
        sc = self.ax.scatter([], [], 
                        s=self.marker_size, 
                        color=color,
                        edgecolor=self.marker_edge_color, 
                        linewidth=self.marker_edge_width,
                        transform=ccrs.PlateCarree(),
                        zorder=10)
        self.scatters.append(sc)
        txt = self.ax.text(0, 0, "", 
                      transform=ccrs.PlateCarree(),
                      fontsize=6,
                      color='black',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
        self.texts.append(txt)
#        trail, = self.ax.plot([], [], 
#                         alpha=0.4, 
#                         linewidth=2, 
#                         transform=ccrs.PlateCarree())
#        self.trails.append(trail)

# Animation update function
    def update(self, frame):
        self.current_frame = frame
        for index in range(len(self.satellite_data)):
            satellite = self.satellite_data[index]
            # Calculate progress (0 to 1)

            # Linear interpolation between start and end
            current_lat = satellite.get_lat_lon_by_index(frame)[0]
            current_lon = satellite.get_lat_lon_by_index(frame)[1]
            
            # Update marker position
            self.scatters[index].set_offsets([[current_lon, current_lat]])
            if satellite.is_in_fov_by_index(frame):
                self.scatters[index].set_facecolor('cyan')  # Highlight if in FOV
            else:
                color = self.get_facecolor_by_inclination(satellite.get_inclination())
                self.scatters[index].set_facecolor(color)  # Reset to original color
            
            # Update label position (slightly offset)
            self.texts[index].set_position((current_lon + 3, current_lat + 2))
            self.texts[index].set_text(satellite.get_satellite_name())
            
            # Update trail (show path from start to current position)
            #trail_lons = np.linspace(satellite.get_lat_lon_by_index(0)[1], current_lon, max(2, int(progress * 30)))
            #trail_lats = np.linspace(satellite.get_lat_lon_by_index(0)[0], current_lat, max(2, int(progress * 30)))
            #self.trails[index].set_data(trail_lons, trail_lats)
        
        # Optional: Update title with frame info
        time_str = datetime.fromtimestamp(self.times[frame], tz=timezone.utc).strftime("'%Y-%m-%dT%H:%M UTC'")
        self.title.set_text(f"World Map - Animated Moving Points (Time {time_str})")
        return self.scatters + self.texts

    def show(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.update, 
                    frames=len(self.satellite_data[0].get_times()),
                    interval=self.interval_ms, 
                    blit=False,           # blit=False is safer with Cartopy
                    repeat=self.repeat)

        self.fig.tight_layout()
        plt.show()
        #self.save_animation()  # Save after showing to ensure animation is created

    def pause_animation(self):
        if self.ani is not None:
            if self.is_paused:
                self.ani.event_source.start()
            else:
                self.ani.event_source.stop()
            self.is_paused = not self.is_paused

    def reset_animation(self):
        if self.ani is not None:
            self.current_frame = 0
            self.update(0)  # Reset to first frame
            self.fig.canvas.draw_idle()  # Redraw the canvas to show the reset immediately
    
    def move_forward(self):
        if self.ani is not None and self.current_frame < len(self.satellite_data[0].get_times()) - 1:
            self.current_frame += 1
            self.update(self.current_frame)
            self.fig.canvas.draw_idle()  # Redraw the canvas to show the update immediately
    
    def move_backward(self):
        if self.ani is not None and self.current_frame > 0:
            self.current_frame -= 1
            self.update(self.current_frame)
            self.fig.canvas.draw_idle()  # Redraw the canvas to show the update immediately


# Optional: Save as GIF or MP4 (uncomment if needed)
# ani.save('world_points_animation.gif', writer='pillow', fps=30)

    def on_key(self, event):
        if event.key == ' ':
            self.pause_animation()
        elif event.key == 'r':
            self.reset_animation()
        elif event.key == 'right':
            self.move_forward()
        elif event.key == 'left':
            self.move_backward()

    def save_animation(self, filename='world_points_animation.mp4', fps=30):
        if self.ani is not None:
            self.ani.save(filename, writer='ffmpeg', fps=fps)
            print(f"✅ Animation saved as '{filename}'")
        else:
            print("⚠️ No animation to save. Please call show() first to create the animation.")
    