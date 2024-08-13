file format conventions for point mass balance data
----------------------------------------------------------

file-name convention:
   <glaciername>_annual.dat          : annual point mass balance
   <glaciername>_winter.dat          : winter point mass balance
   <glaciername>_intermediate.dat    : intermediate values of point mass balance  

   <glaciername> : short VAW working name (no special characters)

File-structure (ASCII):
   header-block (4 lines, # at begin, fields separated by ';'):
      (1) data-type, gl-name, gl-number, type
      (2) column headers
      (3) units (yyyymmdd,hhmm,yyyymmdd),hhmm,d.d,m,m,m a.s.l.,cm,kg m-3, mm w.e.)
      (4) analysis / copyright: source institution, revision year(date), bibliographic ref, url
   data-block with 22 columns :
     name; date0; time0; date1; time1; period; date_quality; x_pos; y_pos; z_pos; position_quality;
       mb_raw; density; density_quality; mb_we; measurement_quality; measurement_type; mb_error; 
       reading_err; density_err; error_evaluation_method; source

   fortran-format-string:
     data-block   '(a11,2x,a8,x,a,2x,a8,x,a,f7.1,i3,2x,2f9.1,f8.1,i4,2i6,i4,i7,2i4,3i6,i4,2x,a)'

   unknown / no-data value:
     id                            :  0
     date0, date1, time0, time1    :  00000000, 0000
     all other columns             :  NaN

   fields/columns description with units:

     data-type :  Mass Balance
     gl-name   :  Glacier name  (official spelling in iso-latin)
     gl-number :  VAW glacier id
     type      :  annual point measurement, winter point measurement, intermediate point measurement
     revision  :  year(date) of last revision  [yyyy(mmdd)]

     name                     :  name of measurement (stake, sounding, probing, etc.)
     date0                    :  date of begin of period  [yyyymmdd]
     time0                    :  time of begin of period  [hh]
     date1                    :  date of end of period  [yyyymmdd]
     time1                    :  time of end of period  [hh]
     period                   :  period length [d]
     date_quality             :  quality identifier for date (definitions see below)
     x_pos                    :  x-position of stake (CH1903)
     y_pos                    :  y-position of stake (CH1903)
     z_pos                    :  elevation of stake
     position_quality         :  quality identifier for position (definitions see below)
     mb_raw                   :  raw mass balance measurement [cm]
     density                  :  snow/firn/ice density [kg m-3]
     density_quality          :  quality identifier for density (definitions see below)
     mb_we                    :  point mas balance [mm w.e. / kg m-2]	
     measurement_quality      :  quality identifier for stake reading (definitions see below)
     measurement_type         :  type of mass balance observation (definitions see below)
     mb_error                 :  Uncertainty of point mass balance as square root of the sum of squares of the fractional uncertainties of Density and Raw Balance [mm w.e.]
     reading_error            :  Reading uncertainty of point mass balance [mm w.e.]
     density_error            :  Density uncertainty of point mass balance [mm w.e.]
     error_evaluation_method  :  Method to evaluate uncertainty in point mass balance (definitions see below)
     source                   :  source/observer (definitions see below)



   date_quality:   specifying the accuracy of the observation date    date uncertainty [days]	date error [mm w.e.]
     0: start and end dates estimated/unknown                         20 d;      <2500m a.s.l.: 349 mm w.e.; 2500-3000m: 239 mm w.e.; >3000m: 154 mm w.e.
     1: start and end dates exactly known                             0; 0
     2: start date exactly known, end date estimated/unknown          14 d;      <2500m a.s.l.: 247 mm w.e.; 2500-3000m: 169 mm w.e.; >3000m: 109 mm w.e.
     3: start date estimated/unknown, end date exactly known          14 d;      <2500m a.s.l.: 247 mm w.e.; 2500-3000m: 169 mm w.e.; >3000m: 109 mm w.e.



   position_quality:   specifying the accuracy of the positioning (position refers to end of period)
                                                                                location uncertainty [m]
     0: undefined/unknown                                                       100 m
     1: measured by dGPS                                                          0 m	
     2: measured by handheld GPS                                                  5 m
     3: measured using an alternative method (e.g. theodolite, triangulation)     0 m
     4: estimated from previous measurements                                     20 m
     5: estimated based on altitude information                                 200 m


   density_quality:  specifying the accuracy of density information		        density uncertainty [%]
     (numbers given always refer average density of ablated/accumulated layer, i.e. bulk density)
     0: quality/source unknown                                                          12 %
     1: Ice density                                                                    1.5 %
     2: Measured snow/firn/ice density                                                   5 %
     3: Density of snow/firn estimated from nearby measurements                          8 % 
     4: Density of snow/firn estimated without nearby measurements                      12 %
     5: water equilvalent based on combination of fresh snow density and ice density    10 %  
     6: Estimated based on linear regression / average. 
        Winter: regression with DOY (d), elevation (m), and snow depth (cm); Annual: overall average density (539 kg m-3)	
	(rho = 257.662 + 1.2169*DOY -0.0085*elevation + 0.1394*snow_depth)                  12 %

   measurement_quality:   specifying general accuracy of mass balance observation (qualitative)	    quality error [cm] (additional error contributing to reading uncertainty)
     0: quality/source unknown                                                                      25 cm
     1: typical reading uncertainty                                                                  0 cm
     2: high reading uncertainty (e.g. stake bent)                                                  20 cm
     3: reconstructed value/exceeds minimum measurement range (e.g. stake completely melted-out)    40 cm
     4: reconstructed value/exceeds maximum measurement range (e.g. stake buried by snow)           40 cm
     5: reconstructed value (other reason)                                                          40 cm


   measurement_type:   specifying the type of mass balance measurement                   type error [cm]
     0: unknown                                                                          10 cm
     1: stake                                                                             5 cm
     2: depth probing / snowpit / coring                                                 10 cm
     3: marked horizon (eg. in snowpit or coring)                                         5 cm
     4: ground-penetrating radar                                                          5% of snow depth, min 10 cm
     5: snowline                                                                          -
     6: nivometer (painted marks on rock face)                                          150 cm
     7: Holfuy Cameras                                                                    2 cm 
     8: other                                                                            30 cm


   error_evaluation_method:    Method used to evaluate uncertainty in point mass balance             
     0: unknown, but estimated
     1: evaluated directly from the measurement procedure
     2: evaluated based on results of nearby measurements (same day, same glacier)


   # sources:
     NN: unknown
     glrep: Glaciological Reports
     firep: Firnberichte
     vaw: documents stored at VAW-ETHZ
     vaw-nf: documents previously stored at VAW-ETHZ but could not be located anymore
     kwm: Kraftwerke Mattmark
     merc/plm: Mercanton 1916, Vermessungen am Rhonegletscher
     pn: Pro Natura
     uzh: Universitaet Zuerich
     unil: Uni Lausanne
     PSI: Paul Scherer Institute
     ...
     ab: Andreas Bauder
     mh: Matthias Huss
     al: Andreas Linsbauer
     rm: Raphi Moser
     mf: Mauro Fischer
     mfu/fu: Martin Funk
	 boe: Hermann Bösch
	 sm: Willy Schmid
     gka: Giovanni Kappenberger
     ust: Urs Steinegger
     mt: Michael Thalmann
     pb: Peter Beglinger
     ol: Otto Langenegger
     mz: Michael Zemp
     sn: Samuel Nussbaumer
     hm: Horst Machguth
     ns: Nadine Salzmann
     jo: Hans Oerlemanns
     hz: Harry Zekollari
     ph: Philippe Huybrechts
     lvt: Lander van Tricht
     av: Andreas Vieli
     bm: Boris Mueller
     df: Daniel Farinotti
     jl: Johannes Landmann
     co: Christophe Ogier
     cl: Christophe Lambiel
     

