# TLE Class
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import skyfield
from skyfield.api import load, wgs84, EarthSatellite
from datetime import datetime, timedelta, timezone

class TLE:
    TLE_FIELDS = ['name', 'norad_id', 'classification', 'intl_designator', 'epoch_year', 'epoch_day', 'mean_motion_dt', 
                  'mean_motion_ddt', 'bstar', 'ephemeris_type', 'element_number', 'inclination', 'raan', 'eccentricity', 
                  'arg_perigee', 'mean_anomaly', 'mean_motion', 'rev_number']

    # Global variable for TLE Start Time and Duration
    TLE_START_TIME: datetime = datetime.now(timezone.utc)
    TLE_END_TIME: datetime = TLE_START_TIME + timedelta(hours=2, minutes=0)
    TLE_STEPS_SECONDS: int = 60
    TLE_DEFAULT_FOV_ANGLE_DEG: float = 5.0



    def __init__(self):
        self.sat_name: str = ""
        # tle_line1 and tle_line2 are the two lines of TLE data that contain the satellite's orbital parameters
        self.tle_line1: str = ""
        self.tle_line2: str = ""
        self.tle_object: object = {}
        self.geocentric: object = None
        self.start_time: datetime = TLE.TLE_START_TIME
        self.end_time: datetime = TLE.TLE_END_TIME
        self.steps_seconds: int = TLE.TLE_STEPS_SECONDS
        self.times: list = []
        self.satellite: EarthSatellite = None
        self.foot_print_radius_km: list = None
        self.default_fov_angle_deg: float = TLE.TLE_DEFAULT_FOV_ANGLE_DEG
        self.fov_intercepts: dict = None

    # Create getters and setters for each the TLE data element in TLE_FIELDS
    def get_satellite_name(self) -> str:
        return self.sat_name

    def set_default_fov_angle(self, fov_angle_deg: float) -> None:
        self.default_fov_angle_deg = fov_angle_deg

    def get_footprint_radius_km(self) -> list:
        if self.foot_print_radius_km is None:
            self.generate_ground_track()

        return self.foot_print_radius_km

    def set_satellite_name(self, sat_name: str) -> None:
        self.sat_name = sat_name

    def get_norad_id(self) -> int:
        return int(self.tle_object.get(TLE.TLE_FIELDS[1], 0))
    
    def set_norad_id(self, norad_id: int) -> None:
        self.tle_object[TLE.TLE_FIELDS[1]] = norad_id  

    def get_classification(self) -> str:
        return self.tle_object.get(TLE.TLE_FIELDS[2], "").strip()
    
    def get_intl_designator(self) -> str:
        return self.tle_object.get(TLE.TLE_FIELDS[3], "").strip()  
    
    def get_epoch_year(self) -> int:
        return int(self.tle_object.get(TLE.TLE_FIELDS[4], 0))
    
    def get_epoch_day(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[5], 0.0))
    
    def get_mean_motion_dt(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[6], 0.0))
    
    def get_mean_motion_ddt(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[7], 0.0))
    
    def get_bstar(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[8], 0.0))
    
    def get_ephemeris_type(self) -> int:
        return int(self.tle_object.get(TLE.TLE_FIELDS[9], 0))
    
    def get_element_number(self) -> int:
        return int(self.tle_object.get(TLE.TLE_FIELDS[10], 0))
    
    def get_inclination(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[11], 0.0))
    
    def get_raan(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[12], 0.0))
    
    def get_eccentricity(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[13], 0.0))
    
    def get_arg_perigee(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[14], 0.0))
    
    def get_mean_anomaly(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[15], 0.0))
    
    def get_mean_motion(self) -> float:
        return float(self.tle_object.get(TLE.TLE_FIELDS[16], 0.0))
    
    def get_rev_number(self) -> int:
        return int(self.tle_object.get(TLE.TLE_FIELDS[17], 0))

    def fixed_width_string(self, text: str, width: int, align: str = 'left', fill_char: str = ' ') -> str:
        """
        Convert a string to exactly 'width' characters.
        
        Parameters:
            text      : The string to format
            width     : Desired total length
            align     : 'left', 'right', or 'center'
            fill_char : Character to pad with (default is space)
        
        Returns:
            String exactly 'width' characters long
        """
        text = str(text)  # ensure it's a string
        
        if len(text) > width:
            return text[:width]          # truncate if too long
        
        if align == 'left':
            return text.ljust(width, fill_char)
        elif align == 'right':
            return text.rjust(width, fill_char)
        elif align == 'center':
            return text.center(width, fill_char)
        else:
            raise ValueError("align must be 'left', 'right', or 'center'")

    def tle_name(self,name: str) -> str:
        """Satellite name: exactly 24 characters, left-aligned"""
        return self.fixed_width_string(name, 24, 'left')

    def tle_int(self, value: int, width: int) -> str:
        """Integer field, right-aligned"""
        return self.fixed_width_string(value, width, 'right')
    
    def tle_float(self, value: str, width: int, decimals: int = None) -> str:
        """Float with fixed width (common in TLE)"""
        try:
            value = float(value)
        except ValueError:
            if '+' in value or '-' in value:
            # Find the last + or - (the exponent sign)
                for i in range(len(value)-1, 0, -1):
                    if value[i] in '+-':
                        # Insert 'e' before the exponent
                        value = value[:i] + 'e' + value[i:]
                        break
        value = float(value)
        if decimals is not None:
            s = f"{value:.{decimals}f}"
        else:
            s = str(value)
        return self.fixed_width_string(s, width, 'right')

    # Method to parse line1 and line2 into TLE data elements
    def parse_tle_from_data(self, sat_name: str, tle_line1: str, tle_line2: str ) -> None:
        self.sat_name = sat_name
        # tle_line1 and tle_line2 are the two lines of TLE data that contain the satellite's orbital parameters
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.parse_tle_to_dict()
    
    def parse_tle_to_dict(self, sat_name: str=None, tle_line1: str=None, tle_line2: str=None) -> None:
        # Implementation for parsing TLE data to dictionary format
        if sat_name is not None:
            self.sat_name = sat_name
        if tle_line1 is not None:
            self.tle_line1 = tle_line1
        if tle_line2 is not None:
            self.tle_line2 = tle_line2

        self.tle_object[TLE.TLE_FIELDS[0]] = self.sat_name
        # Parse tle_line1 and tle_line2 to extract the TLE data elements and store them in the tle_object dictionary
        
        self.tle_object[TLE.TLE_FIELDS[1]] = self.tle_line1[2:7]  # norad_id
        self.tle_object[TLE.TLE_FIELDS[2]] = self.tle_line1[7:8]  # classification
        self.tle_object[TLE.TLE_FIELDS[3]] = self.tle_line1[8:17]  # intl_designator
        self.tle_object[TLE.TLE_FIELDS[4]] = self.tle_line1[17:20]  # epoch_year
        self.tle_object[TLE.TLE_FIELDS[5]] = self.tle_line1[20:32]  # epoch_day
        self.tle_object[TLE.TLE_FIELDS[6]] = self.tle_line1[32:43]  # mean_motion_dt
        self.tle_object[TLE.TLE_FIELDS[7]] = self.tle_line1[43:52]  # mean_motion_ddt
        self.tle_object[TLE.TLE_FIELDS[8]] = self.tle_line1[52:61]  # bstar
        self.tle_object[TLE.TLE_FIELDS[9]] = self.tle_line1[61:63]  # ephemeris_type
        self.tle_object[TLE.TLE_FIELDS[10]] = self.tle_line1[63:69]  # element_number
        self.tle_object[TLE.TLE_FIELDS[11]] = self.tle_line2[7:16]  # inclination
        self.tle_object[TLE.TLE_FIELDS[12]] = self.tle_line2[16:25]  # raan
        self.tle_object[TLE.TLE_FIELDS[13]] = self.tle_line2[25:33]  # eccentricity
        self.tle_object[TLE.TLE_FIELDS[14]] = self.tle_line2[33:42]  # arg_perigee
        self.tle_object[TLE.TLE_FIELDS[15]] = self.tle_line2[42:51]  # mean_anomaly
        self.tle_object[TLE.TLE_FIELDS[16]] = self.tle_line2[51:63]  # mean_motion
        self.tle_object[TLE.TLE_FIELDS[17]] = self.tle_line2[63:69]  # rev_number

    def parse_tle_from_json(self, json_str: str) -> None:
        # Implementation for parsing TLE data from JSON stringtle_dict.get(TLE.TLE_FIELDS[5]
        tle_dict = json.loads(json_str)
        self.parse_tle_from_dict(tle_dict)


    def parse_tle_from_dict(self, tle_dict: dict) -> None:
        # Implementation for parsing TLE data from JSON data
        self.sat_name = tle_dict.get(TLE.TLE_FIELDS[0], "")
        self.tle_line1 = "1 "
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[1], "")                    # norad_id
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[2], " ")                               # classification
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[3], "")                                # intl_designator
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[4], "")                                # epoch_year
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[5], "")              # epoch_day
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[6], "")        # mean_motion_dt
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[7], "")              # mean_motion_ddt
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[8], "")              # bstar
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[9], "")                    # ephemeris_type
        self.tle_line1 += tle_dict.get(TLE.TLE_FIELDS[10], "")                   # element_number
        self.tle_line2 = "2 "
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[1], "")                    # norad_id
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[11], "")              # inclination
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[12], "")              # raan
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[13], "")              # eccentricity
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[14], "")              # arg_perigee
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[15], "")              # mean_anomaly
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[16], "")             # mean_motion
        self.tle_line2 += tle_dict.get(TLE.TLE_FIELDS[17], "")                   # rev_number

        self.parse_tle_to_dict()

    def get_data(self) -> dict:
        return {
            "sat_name": self.sat_name,
            "tle_line1": self.tle_line1,
            "tle_line2": self.tle_line2
        }
    
    def get_as_json_string(self) -> str:
        return json.dumps(self.tle_object, indent=4)
    
    def get_data_as_dict(self) -> dict:
        return self.tle_object

    
    def generate_ground_track(self) -> None:
        # Implementation for generating ground track data (latitude, longitude, altitude) over time
        ts = load.timescale()
        t0 = ts.utc(self.start_time)
        t1 = ts.utc(self.end_time)
        # Create evenly spaced times (every steps_seconds seconds)
        numpts = int((t1.utc_datetime() - t0.utc_datetime()).total_seconds() / self.steps_seconds) + 1
        self.times = ts.linspace(t0, t1, num=numpts)

        # ==================== COMPUTE POSITIONS ====================
        self.satellite = EarthSatellite(self.tle_line1, self.tle_line2, self.sat_name, ts)

        # Get geocentric position at all times
        self.geocentric = self.satellite.at(self.times)
        heights = wgs84.height_of(self.geocentric).km
        # For each element in heights, calculate the footprint radius in kilometers using the function self.calculate   _footprint and store the results in self.foot_print_radius_km
        self.foot_print_radius_km = [ self.calculate_footprint(altitude_km=h) for h in heights ]   

    def get_satellite_info(self) -> dict: 
        # Convert to latitude, longitude, and height (km)
        if self.geocentric is None:
            self.generate_ground_track()
        statellite_info = {}
        statellite_info["time_hr"] = [t.utc_datetime().isoformat() for t in self.times]
        statellite_info["time"] = [ int(t.utc_datetime().timestamp()) for t in self.times]    
        statellite_info["latitude"] = wgs84.latlon_of(self.geocentric)[0].degrees
        statellite_info["longitude"] = wgs84.latlon_of(self.geocentric)[1].degrees
        statellite_info["height_km"] = wgs84.height_of(self.geocentric).km
        statellite_info["footprint_radius_km"] = self.foot_print_radius_km

        return statellite_info


    def plot_Latitude(self, ax: plt.Axes = None) -> None:
        if self.geocentric is None:
            self.generate_ground_track()
        latitudes = wgs84.latlon_of(self.geocentric)[0].degrees
        # Set the x-axis to show time in UTC format and the y-axis to show latitude from +90 to -90 in degrees, and add grid lines
        
        if ax is None:
            plt.figure(figsize=(10, 5))
            plt.plot(latitudes)
            plt.xlim(0, len(latitudes) - 1)
            plt.ylim(-100, 100)
            #set title font size to 10
            if plt.gca().get_title() == "":
                plt.title(f"Latitude vs Time for\n{self.sat_name}", fontsize='small')
            else:
                plt.title(plt.gca().get_title() + f", {self.sat_name}", fontsize='small')
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Latitude (degrees)")
            plt.grid(True)
            plt.show()
        else:
            ax.plot(latitudes)
            ax.set_xlim(0, len(latitudes) - 1)
            ax.set_ylim(-100, 100)
            if ax.get_title() == "":
                ax.set_title(f"Latitude vs Time for\n{self.sat_name}", fontsize='small')
            else:
                ax.set_title(ax.get_title() + f", {self.sat_name}", fontsize='small')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Latitude (degrees)")
            ax.grid(True)

    def plot_Longitude(self, ax: plt.Axes = None) -> None:
        if self.geocentric is None:
            self.generate_ground_track()
        longitudes = wgs84.latlon_of(self.geocentric)[1].degrees
        # Set the x-axis to show time in UTC format and the y-axis to show longitude from +180 to -180 in degrees, and add grid lines
        if ax is None:
            plt.figure(figsize=(10, 5))
            plt.plot(longitudes)
            plt.xlim(0, len(longitudes) - 1)
            plt.ylim(-200, 200)
            if plt.gca().get_title() == "":
                plt.title(f"Longitude vs Time for\n{self.sat_name}", fontsize='small')
            else:
                plt.title(plt.gca().get_title() + f", {self.sat_name}", fontsize='small')
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Longitude (degrees)")
            plt.grid(True)
            plt.show()
        else:
            ax.plot(longitudes)
            ax.set_xlim(0, len(longitudes) - 1)
            ax.set_ylim(-200, 200)
            if ax.get_title() == "":
                ax.set_title(f"Longitude vs Time for\n{self.sat_name}", fontsize='small')
            else:
                ax.set_title(ax.get_title() + f", {self.sat_name}", fontsize='small')
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Longitude (degrees)")
            ax.grid(True)

    def plot_Altitude(self, ax: plt.Axes = None) -> None:
        if self.geocentric is None:
            self.generate_ground_track()
        altitudes = wgs84.height_of(self.geocentric).km

        # Set the x-axis to show time in UTC format and the y-axis to show altitude in kilometers, and add grid lines
        if ax is None:
            plt.figure(figsize=(10, 5))
            plt.plot(altitudes)
            plt.xlim(0, len(altitudes) - 1)
            if plt.gca().get_title() == "":
                plt.title(f"Altitude vs Time for\n{self.sat_name}", fontsize='small')
            else:
                plt.title(plt.gca().get_title() + f", {self.sat_name}", fontsize='small')
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Altitude (km)")
            plt.grid(True)
            plt.show()
        else:
            ax.plot(altitudes)
            ax.set_xlim(0, len(altitudes) - 1)
            if ax.get_title() == "":
                ax.set_title(f"Altitude vs Time for\n {self.sat_name}", fontsize='small')
            else:
                ax.set_title(ax.get_title() + f", {self.sat_name}", fontsize='small')
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Altitude (km)")
            ax.grid(True)
    
    def plot_ground_track(self, ax: plt.Axes = None) -> None:
        if self.geocentric is None:
            self.generate_ground_track()
        latitudes = wgs84.latlon_of(self.geocentric)[0].degrees
        longitudes = wgs84.latlon_of(self.geocentric)[1].degrees

        # Set the x-axis to show longitude from +180 to -180 in degrees and the y-axis to show latitude from +90 to -90 in degrees, and add grid lines
        # Make the legend smaller and place it in the upper right corner
        if ax is None:
            plt.figure(figsize=(10, 5))
            plt.scatter(longitudes, latitudes, s=1, label=self.sat_name)
            plt.scatter(longitudes[0], latitudes[0], color='green', s=5, label='Start')
            plt.scatter(longitudes[-1], latitudes[-1], color='red', s=5, label='End')
            plt.xlim(-190, 190)
            plt.ylim(-100, 100)
            if plt.gca().get_title() == "":
                plt.title(f"Ground Track for\n{self.sat_name}", fontsize='small')
            else:
                plt.title(plt.gca().get_title() + f", {self.sat_name}", fontsize='small')
            plt.xlabel("Longitude (degrees)")
            plt.ylabel("Latitude (degrees)")
            plt.legend(fontsize='small' , loc='upper right')
            plt.grid(True)
            plt.show()
            
        else:
            ax.scatter(longitudes, latitudes, s=1, label=self.sat_name)
            ax.scatter(longitudes[0], latitudes[0], color='green', s=5, label='Start')
            ax.scatter(longitudes[-1], latitudes[-1], color='red', s=5, label='End')
            ax.set_xlim(-190, 190)
            ax.set_ylim(-100, 100)
            if ax.get_title() == "":
                ax.set_title(f"Ground Track for\n{self.sat_name}", fontsize='small')
            else:
                ax.set_title(ax.get_title() + f", {self.sat_name}", fontsize='small')
            ax.set_xlabel("Longitude (degrees)")
            ax.set_ylabel("Latitude (degrees)")
            #ax.legend(fontsize='small' , loc='upper right')
            ax.grid(True)

    # Method to plot the distance between two TLE objects at a given time range in kilometers
    def plot_distance_to_other_satellite(self, other_tle: object, ax: plt.Axes=None, time_range: list=None, in_view_only: bool=False) -> None:
        # Implementation for plotting distance between two TLE objects at a given time range
        distances = self.distance_to_other_satellite(other_tle, time_range, in_view_only=in_view_only)

        if ax is None:
            plt.figure(figsize=(10, 5))
            plt.plot(distances["distance_km"], label=f"Distance between {self.sat_name} and {other_tle.sat_name}")
        else:
            ax.plot(distances["distance_km"], label=f"Distance between {self.sat_name} and {other_tle.sat_name}")
        if ax is None:
            plt.xlabel("Time (Seconds)")
            plt.ylabel("Distance (km)")
            plt.title(f"Distance between {self.sat_name} and {other_tle.sat_name} vs Time", fontsize='small')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend(fontsize='small' , loc='upper right')
            plt.tight_layout()
            plt.show()
        else:
            ax.set_xlabel("Time (Seconds)")
            ax.set_ylabel("Distance (km)")
            ax.set_title(f"Distance between {self.sat_name} and {other_tle.sat_name} vs Time", fontsize='small')
            ax.set_xticks(rotation=45)
            ax.grid(True)
            ax.legend(fontsize='small' , loc='upper right')
            ax.tight_layout()

    def get_distance_to_other_satellite(self, other_tle: object, time: float) -> float:
        pos1 = self.satellite.at(time).xyz.km
        pos2 = other_tle.satellite.at(time).xyz.km
        return (pos1 - pos2).distance().km

    def distance_to_other_satellite(self, other_tle: object, time_range: list=None, in_view_only: bool=False) -> dict:
        # Implementation for calculating distance between two TLE objects at a given time range
        
        if self.geocentric is None:
            self.generate_ground_track()

        if other_tle.geocentric is None:
            other_tle.generate_ground_track()

        if time_range is None:
            time_range = self.times
        
        # Calculate the distance between the two TLE objects at each time step and return the average distance in kilometers
        distances = {}
        distances["satellite1"] = self.sat_name
        distances["satellite2"] = other_tle.sat_name
        distances["time"] = []
        distances["time_hr"] = []
        distances["distance_km"] = []
        
        index = 0
        for t in time_range:
            distances["time_hr"].append(t.utc_datetime().isoformat())
            distances["time"].append(t)

            
            
            pos1 = self.satellite.at(time_range[index])
            pos2 = other_tle.satellite.at(other_tle.times[index])
            
            in_view = self.in_view_of_other_satellite(other_tle, time_range[index])

            if not in_view and in_view_only:
                distance = float(-1)  # or some large number to indicate they are not visible to each other            
            else:
                distance = self.get_distance_to_other_satellite(other_tle, time_range[index])       
            distances["distance_km"].append(distance)
            index += 1
        
        return distances
    
    # Method to determine if two TLE objects are in view of each other at a given time
    def in_view_of_other_satellite(self, other_tle: object, time: float) -> bool:
        # Implementation for determining if two TLE objects are in view of each other at a given time
        
        if self.geocentric is None:
            self.generate_ground_track()

        if other_tle.geocentric is None:
            other_tle.generate_ground_track()
        # Calculate the distance between the two TLE objects at the given time
        pos1 = self.satellite.at(time).xyz.km
        pos2 = other_tle.satellite.at(time).xyz.km

        # Calculate the vector from Sat1 to Sat2
        vector_from_sat1_to_sat2 = pos2 - pos1

        # Check if the Earth blocks the line of sight from Sat1's perspective
        cross = np.cross(pos1, vector_from_sat1_to_sat2)
        perp_distance = np.linalg.norm(cross) / np.linalg.norm(vector_from_sat1_to_sat2)

        radius_earth_km = 6371  # Average radius of Earth in kilometers
        in_view = perp_distance > radius_earth_km
        return in_view
    
    # Method to calculate the footprint of a TLE object given its altitude and a field of view angle in degrees
    def calculate_footprint(self, altitude_km: float, fov_angle_deg: float = None) -> float:
        if fov_angle_deg is None:
            fov_angle_deg = self.default_fov_angle_deg
        
        # Calculate the footprint radius in kilometers based on the altitude and field of view angle
        fov_angle_rad = np.radians(fov_angle_deg)
        footprint_radius_km = altitude_km * np.tan(fov_angle_rad / 2)

        return footprint_radius_km
    
    # Method to determine if the fov of a TLE object overlaps with the fov of another TLE object at a given time
    def fov_overlaps_with_other_satellite(self, other_tle: object) -> None:
        # Implementation for determining if the fov of a TLE object overlaps with the fov of another TLE object at a given time
        if self.geocentric is None:
            self.generate_ground_track()

        if other_tle.geocentric is None:
            other_tle.generate_ground_track()
        
        for index in range(0, len(self.times)):
            t = self.times[index]
            fov = {}
            fov["time"] = t
            fov["time_hr"] = t.utc_datetime().isoformat()
            fov["other_satellite_name"] = other_tle.sat_name
            fov["in_view"] = self.in_view_of_other_satellite(other_tle, t)
            fov_range_km = self.foot_print_radius_km[index] + other_tle.foot_print_radius_km[index]
            distance_between_sats_km = self.get_distance_to_other_satellite(other_tle, t)
            fov["fov_overlap"] = distance_between_sats_km < fov_range_km 
            fov["fov_overlap_km"] = fov_range_km
            
            self.add_fov_intercept(fov)


    def add_fov_intercept(self, intercept: dict) -> None:
        if self.fov_intercepts is None:
            self.fov_intercepts = {}
        # Check if the name of the other satellite is in the fov_intercepts dictionary, if not add it with an empty list
        if dict["other_satellite_name"] not in self.fov_intercepts.keys():
            self.fov_intercepts[dict["other_satellite_name"]] = []
        self.fov_intercepts[dict["other_satellite_name"]].append(intercept)
