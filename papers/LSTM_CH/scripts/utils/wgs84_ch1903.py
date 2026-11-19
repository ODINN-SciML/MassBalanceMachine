#!/usr/bin/python2
# -*- coding: utf-8 -*-

#  The MIT License (MIT)
#
#  Copyright (c) 2014 Federal Office of Topography swisstopo, Wabern, CH and Aaron Schmocker
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#

# WGS84 <-> LV03 converter based on the scripts of swisstopo written for python2.7
# Aaron Schmocker [aaron@duckpond.ch]
# vim: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

# Source: http://www.swisstopo.admin.ch/internet/swisstopo/en/home/topics/survey/sys/refsys/projections.html (see PDFs under "Documentation")
# Updated 9 dec 2014
# Please validate your results with NAVREF on-line service: http://www.swisstopo.admin.ch/internet/swisstopo/en/home/apps/calc/navref.html (difference ~ 1-2m)

"""Approximate conversion between Swiss LV03 (CH1903) and WGS84 coordinates.

Uses the swisstopo approximate formulas (accuracy ~1 m); see the license and
source notes above.
"""

import math


class GPSConverter(object):
    """
    GPS Converter class which is able to perform convertions between the
    CH1903 and WGS84 system.

    Note the Swiss convention used in the method names: ``y`` is the easting
    and ``x`` the northing (in metres, LV03). Angles are in decimal degrees.
    """

    def CHtoWGSheight(self, y, x, h):
        """Convert CH y/x/h to WGS height."""
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        h = (h + 49.55) - (12.60 * y_aux) - (22.64 * x_aux)
        return h

    def CHtoWGSlat(self, y, x):
        """Convert CH y/x to WGS lat."""
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lat = (
            (16.9023892 + (3.238272 * x_aux))
            + -(0.270978 * pow(y_aux, 2))
            + -(0.002528 * pow(x_aux, 2))
            + -(0.0447 * pow(y_aux, 2) * x_aux)
            + -(0.0140 * pow(x_aux, 3))
        )
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lat = (lat * 100) / 36
        return lat

    def CHtoWGSlng(self, y, x):
        """Convert CH y/x to WGS long."""
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lng = (
            2.6779094
            + (4.728982 * y_aux)
            + +(0.791484 * y_aux * x_aux)
            + +(0.1306 * y_aux * pow(x_aux, 2))
        ) + -(0.0436 * pow(y_aux, 3))
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lng = (lng * 100) / 36
        return lng

    def DecToSexAngle(self, dec):
        """Convert decimal angle (° dec) to sexagesimal angle (dd.mmss,ss)."""
        degree = int(math.floor(dec))
        minute = int(math.floor((dec - degree) * 60))
        second = (((dec - degree) * 60) - minute) * 60
        return degree + (float(minute) / 100) + (second / 10000)

    def SexAngleToSeconds(self, dms):
        """Convert sexagesimal angle (dd.mmss,ss) to seconds."""
        degree = 0
        minute = 0
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return second + (minute * 60) + (degree * 3600)

    def SexToDecAngle(self, dms):
        """Convert sexagesimal angle (dd.mmss) to decimal angle (degrees)."""
        degree = 0
        minute = 0
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return degree + (minute / 60) + (second / 3600)

    def WGStoCHh(self, lat, lng, h):
        """Convert WGS lat/long (° dec) and height to CH h."""
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        h = (h - 49.55) + (2.73 * lng_aux) + (6.94 * lat_aux)
        return h

    def WGStoCHx(self, lat, lng):
        """Convert WGS lat/long (° dec) to CH x."""
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        x = (
            (
                200147.07
                + (308807.95 * lat_aux)
                + +(3745.25 * pow(lng_aux, 2))
                + +(76.63 * pow(lat_aux, 2))
            )
            + -(194.56 * pow(lng_aux, 2) * lat_aux)
        ) + +(119.79 * pow(lat_aux, 3))
        return x

    def WGStoCHy(self, lat, lng):
        """Convert WGS lat/long (° dec) to CH y."""
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        y = (
            (600072.37 + (211455.93 * lng_aux))
            + -(10938.51 * lng_aux * lat_aux)
            + -(0.36 * lng_aux * pow(lat_aux, 2))
            + -(44.54 * pow(lng_aux, 3))
        )
        return y

    def LV03_to_WGS84(self, east, north, height):
        """
        Convert LV03 to WGS84 Return a array of double that contain lat, long,
        and height
        """
        d = []
        d.append(self.CHtoWGSlat(east, north))
        d.append(self.CHtoWGSlng(east, north))
        d.append(self.CHtoWGSheight(east, north, height))
        return d

    def WGS84toLV03(self, latitude, longitude, ellHeight):
        """
        Convert WGS84 to LV03 Return an array of double that contaign east,
        north, and height
        """
        d = []
        d.append(self.WGStoCHy(latitude, longitude))
        d.append(self.WGStoCHx(latitude, longitude))
        d.append(self.WGStoCHh(latitude, longitude, ellHeight))
        return d


if __name__ == "__main__":
    """Example usage for the GPSConverter class."""

    converter = GPSConverter()

    # Coordinates
    wgs84 = [46.95108, 7.438637, 0]
    lv03 = []

    # Convert WGS84 to LV03 coordinates
    lv03 = converter.WGS84toLV03(wgs84[0], wgs84[1], wgs84[2])

    print("WGS84: ")
    print(wgs84)
    print("LV03: ")
    print(lv03)
