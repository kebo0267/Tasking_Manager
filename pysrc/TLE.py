# TLE Class
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import skyfield
from skyfield.api import load, wgs84, EarthSatellite
from datetime import datetime, timedelta

class TLE:
    TLE_FIELDS = ['name', 'norad_id', 'classification', 'intl_designator', 'epoch_year', 'epoch_day', 'mean_motion_dt', 
                  'mean_motion_ddt', 'bstar', 'ephemeris_type', 'element_number', 'inclination', 'raan', 'eccentricity', 
                  'arg_perigee', 'mean_anomaly', 'mean_motion', 'rev_number']


    def __init__(self):
        self.sat_name: str = ""
        # tle_line1 and tle_line2 are the two lines of TLE data that contain the satellite's orbital parameters
        self.tle_line1: str = ""
        self.tle_line2: str = ""
        self.tle_object: object = {}
        self.geocentric: object = None
    # Create getters and setters for each the TLE data element in TLE_FIELDS
    def get_sat_name(self) -> str:
        return self.sat_name

    def set_sat_name(self, sat_name: str) -> None:
        self.sat_name = sat_name

    def get_norad_id(self) -> int:
        return self.tle_object.get(TLE.TLE_FIELDS[1], 0)
    
    def set_norad_id(self, norad_id: int) -> None:
        self.tle_object[TLE.TLE_FIELDS[1]] = norad_id  

    def get_classification(self) -> str:
        return self.tle_object.get(TLE.TLE_FIELDS[2], "")
    
    def get_intl_designator(self) -> str:
        return self.tle_object.get(TLE.TLE_FIELDS[3], "")
    
    def get_epoch_year(self) -> int:
        return self.tle_object.get(TLE.TLE_FIELDS[4], 0)
    
    def get_epoch_day(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[5], 0.0)
    
    def get_mean_motion_dt(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[6], 0.0)
    
    def get_mean_motion_ddt(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[7], 0.0)
    
    def get_bstar(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[8], 0.0)
    
    def get_ephemeris_type(self) -> int:
        return self.tle_object.get(TLE.TLE_FIELDS[9], 0)
    
    def get_element_number(self) -> int:
        return self.tle_object.get(TLE.TLE_FIELDS[10], 0)
    
    def get_inclination(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[11], 0.0)
    
    def get_raan(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[12], 0.0)
    
    def get_eccentricity(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[13], 0.0)
    
    def get_arg_perigee(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[14], 0.0)
    
    def get_mean_anomaly(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[15], 0.0)
    
    def get_mean_motion(self) -> float:
        return self.tle_object.get(TLE.TLE_FIELDS[16], 0.0)
    
    def get_rev_number(self) -> int:
        return self.tle_object.get(TLE.TLE_FIELDS[17], 0)

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
    
    def parse_tle_to_dict(self, sat_name: str=None, tle_line1: str=None, tle_line2: str=None) -> dict:
        # Implementation for parsing TLE data to dictionary format
        if sat_name is not None:
            self.sat_name = sat_name
        if tle_line1 is not None:
            self.tle_line1 = tle_line1
        if tle_line2 is not None:
            self.tle_line2 = tle_line2

        self.tle_object[TLE.TLE_FIELDS[0]] = self.sat_name
        # Parse tle_line1 and tle_line2 to extract the TLE data elements and store them in the tle_object dictionary
        
        self.tle_object[TLE.TLE_FIELDS[1]] = int(self.tle_line1[2:7].strip())  # norad_id
        self.tle_object[TLE.TLE_FIELDS[2]] = self.tle_line1[7:8].strip()  # classification
        self.tle_object[TLE.TLE_FIELDS[3]] = self.tle_line1[9:17].strip()  # intl_designator
        self.tle_object[TLE.TLE_FIELDS[4]] = int(self.tle_line1[18:20].strip())  # epoch_year
        self.tle_object[TLE.TLE_FIELDS[5]] = float(self.tle_line1[20:32].strip())  # epoch_day
        self.tle_object[TLE.TLE_FIELDS[6]] = float(self.tle_line1[33:43].strip())  # mean_motion_dt
        self.tle_object[TLE.TLE_FIELDS[7]] = float(self.tle_float(self.tle_line1[44:52].replace(' ','0'), 8, 8))  # mean_motion_ddt
        self.tle_object[TLE.TLE_FIELDS[8]] = float(self.tle_float(self.tle_line1[53:61].replace(' ','0'), 8, 8))  # bstar
        self.tle_object[TLE.TLE_FIELDS[9]] = int(self.tle_line1[62:63].strip())  # ephemeris_type
        self.tle_object[TLE.TLE_FIELDS[10]] = int(self.tle_line1[64:68].strip())  # element_number
        self.tle_object[TLE.TLE_FIELDS[11]] = float(self.tle_line2[8:16].strip())  # inclination
        self.tle_object[TLE.TLE_FIELDS[12]] = float(self.tle_line2[17:25].strip())  # raan
        self.tle_object[TLE.TLE_FIELDS[13]] = float('0.' + self.tle_line2[26:33].strip())  # eccentricity
        self.tle_object[TLE.TLE_FIELDS[14]] = float(self.tle_line2[34:42].strip())  # arg_perigee
        self.tle_object[TLE.TLE_FIELDS[15]] = float(self.tle_line2[43:51].strip())  # mean_anomaly
        self.tle_object[TLE.TLE_FIELDS[16]] = float(self.tle_line2[52:63].strip())  # mean_motion
        self.tle_object[TLE.TLE_FIELDS[17]] = int(self.tle_line2[63:68].strip())  # rev_number

    def parse_tle_from_json(self, json_str: str) -> None:
        # Implementation for parsing TLE data from JSON string
        tle_dict = json.loads(json_str)
        self.parse_tle_from_dict(tle_dict)


    def parse_tle_from_dict(self, tle_dict: dict) -> None:
        # Implementation for parsing TLE data from JSON data
        self.sat_name = tle_dict.get(TLE.TLE_FIELDS[0], "")
        self.tle_line1 = "1 "
        self.tle_line1 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[1], ""), 5)                    # norad_id
        self.tle_line1 += self.tle_dict.get(TLE.TLE_FIELDS[2], " ")                               # classification
        self.tle_line1 += " " + self.fixed_width_string(tle_dict.get(TLE.TLE_FIELDS[3], ""), 8)   # intl_designator
        self.tle_line1 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[4], ""), 2)                    # epoch_year
        self.tle_line1 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[5], ""), 12, 8)              # epoch_day
        self.tle_line1 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[6], ""), 10, 8)              # mean_motion_dt
        self.tle_line1 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[7], ""), 10, 8)              # mean_motion_ddt
        self.tle_line1 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[8], ""), 10, 8)              # bstar
        self.tle_line1 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[9], ""), 1)                    # ephemeris_type
        self.tle_line1 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[10], ""), 4)                   # element_number
        self.tle_line2 = "2 "
        self.tle_line2 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[1], ""), 5)                    # norad_id
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[11], ""), 8, 4)              # inclination
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[12], ""), 8, 4)              # raan
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[13], ""), 7, 7)              # eccentricity
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[14], ""), 8, 4)              # arg_perigee
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[15], ""), 8, 4)              # mean_anomaly
        self.tle_line2 += self.tle_float(tle_dict.get(TLE.TLE_FIELDS[16], ""), 11, 8)             # mean_motion
        self.tle_line2 += self.tle_int(tle_dict.get(TLE.TLE_FIELDS[17], ""), 5)                   # rev_number


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
    
    def generate_ground_track(self, hours: int, minutes: int=0, steps_seconds: int=60) -> None:
        # Implementation for generating ground track data (latitude, longitude, altitude) over time
        ts = load.timescale()
        t0 = ts.now()                                   # current UTC time
        self.generate_ground_track(t0.utc_datetime(), hours, minutes, steps_seconds)

    
    def generate_ground_track(self, start_time: datetime, hours: int, minutes: int=0, steps_seconds: int=60) -> None:
        # Implementation for generating ground track data (latitude, longitude, altitude) over time
        ts = load.timescale()
        t0 = ts.utc(start_time)
        deltaTime = timedelta(hours=hours, minutes=minutes)
        t1 = ts.utc(t0.utc_datetime() + deltaTime)

        # Create evenly spaced times (every steps_seconds seconds)
        numpts = int((t1.utc_datetime() - t0.utc_datetime()).total_seconds() / steps_seconds) + 1
        times = ts.linspace(t0, t1, num=numpts)

        # ==================== COMPUTE POSITIONS ====================
        satellite = EarthSatellite(self.tle_line1, self.tle_line2, self.sat_name, ts)

        # Get geocentric position at all times
        self.geocentric = satellite.at(times)


    def get_lat_lon_alt(self) -> dict: 
        # Convert to latitude, longitude, and height (km)
        if self.geocentric is None:
            raise ValueError("Ground track data not generated yet. Call generate_ground_track() first.")  
        lat_lon_alt = {}
        lat_lon_alt["latitude"] = wgs84.latlon_of(self.geocentric)[0].degrees
        lat_lon_alt["longitude"] = wgs84.latlon_of(self.geocentric)[1].degrees
        lat_lon_alt["height_km"] = wgs84.height_of(self.geocentric).km
        return lat_lon_alt
    

    

    