! MAIN PURPOSE: to forward calculate model state to microwave &
! brightness temperatures (Tbs) by calling the CRTM forward function

! DETAILS ABOUT CRTM: CRTM version 2.3.0 is used with modified &
! coefficient files (work done by Scott Sieron and Yinghui Lv)

! WHAT PART2 IS: Input information of WRF dimensions, and calculates &
! brightness temperature for satellite channels

PROGRAM crtm

    ! ============================================================================
    ! **** ENVIRONMENT SETUP FOR RTM USAGE ****
    ! ============================================================================ 
    
    ! Module usage
    use netcdf
    use mpi_module
    use crtm_module
    use qn2re_wsm6
    use qn2re_gfdlfv3
    use qn2re_thompson08 
    implicit none
   
    ! Configuration
    CHARACTER(*), PARAMETER :: PROGRAM_NAME   = 'crtm'
    CHARACTER(*), PARAMETER :: EXPERIMENT_NAME = 'gpm_gmi_hf'
    CHARACTER(*), PARAMETER :: STORM_NAME = 'Harvey'
    LOGICAL, PARAMETER :: SUPPRESS_OUTPUT = .TRUE.
 
    ! Parameters 
    integer, parameter                       :: ix = 297, jx = 297, kx = 42 !Parameters of WRF domain (ix:number of grids along x direction, jy:number of grids along y direction, kx: number of vertical layers)
    character(256)                           :: inputdir, inputfile, inputWRF 
    character(256)                           :: OUTPUT_DIR
    
    ! character(len=80), intent(in)            :: times
    real                                     :: dx,dy
    
    
    ! Declare WRF variables
    ! ---------------------------------------------
    real :: p(ix,jx,kx)
    real :: pb(ix,jx,kx)
    real :: pres(ix,jx,kx)
    real :: ph(ix,jx,kx+1)
    real :: phb(ix,jx,kx+1)
    real :: t(ix,jx,kx)
    real :: tk(ix,jx,kx)
    real :: qvapor(ix,jx,kx)
    real :: qcloud(ix,jx,kx)
    real :: qrain(ix,jx,kx)
    real :: qice(ix,jx,kx)
    real :: qsnow(ix,jx,kx)
    real :: qgraup(ix,jx,kx)
    real :: psfc(ix,jx)
    real :: hgt(ix,jx)
    real :: tsk(ix,jx)
    real :: xland(ix,jx)
    real :: xlong(ix,jx)
    real :: xlat(ix,jx)
    real :: lat(ix,jx)   ! in radian
    real :: lon(ix,jx)   ! in radian
    real :: u10(ix,jx)
    real :: v10(ix,jx)
    real :: windspeed(ix,jx)
    real :: westwind(ix,jx)
    real :: winddir(ix,jx)
    
    real :: delz(kx)
    real :: nrain(ix,jx,kx)


    ! Declare CRTM variables
    ! ---------------------------------------------  
    INTEGER, PARAMETER :: N_SUPPORTED_SENSOR_CLASSES = 1
    CHARACTER(256) :: Sensor_Id = &
      'gmi_gpm_hf'
      !(/'ssmis','ssmi','gmi_gpm_hf','gmi_gpm_lf','saphir_meghat','amsr2_gcom-w1','atms_npp','atms_n20','mhs'/)
    INTEGER, PARAMETER, DIMENSION(4)  :: channel_subset = &
      (/10,11,12,13/) 
      !(/1,2,3,4,5,6,7,8,9/)  

    !INTEGER, DIMENSION(N_SUPPORTED_SENSOR_CLASSES) :: N_CHANNELS = &
    !  (/13/)
      !(/24, 7, 13, 13, 6, 14, 22, 22, 5/) !total number of channels (of a sensor

    ! zenith angle: gmi_gpm_lf 52.8, gmi_gpm_hf 49.1, ssmis 53.1
    ! scan angle: gmi_gpm_lf 48.5, gmi_gpm_hf 45.4, ssmis 45.0 
    REAL(fp), PARAMETER :: my_zenith_angle = 52.8_fp   
    REAL(fp), PARAMETER :: my_scan_angle = 48.5_fp   
    REAL(fp)            :: my_azimuth_angle

    ! Parameter dimensions in CRTM structures
    INTEGER, PARAMETER :: N_PROFILES  = 1 !Number of atmospheric profile/column
    INTEGER, PARAMETER :: N_ABSORBERS = 2 ! Number of absorbers dimension (must be > 0) 
    INTEGER, PARAMETER :: N_AEROSOLS  = 0 ! Number of aerosols dimension (Can be = 0)
    INTEGER, PARAMETER :: N_SENSORS = 1 ! Number of sensors dimension
    INTEGER, PARAMETER :: N_STREAMS = 16
    integer, parameter :: N_CLOUDS=5 !(if use_cloud is true)
    
    ! Built-in CRTM Variables  
    CHARACTER(256) :: Message
    CHARACTER(256) :: Version
    CHARACTER(256) :: OUTPUT_NAME
    INTEGER :: Error_Status
    INTEGER :: Allocate_Status,DeAllocateStatus
    INTEGER :: n_channels
    real, allocatable, dimension(:,:,:) :: Tbsend, Tb    
    logical :: use_slant_path = .false.
    logical :: use_cloud = .true.
    ! WRF gird-point dimension to CRTM grid-point dimension
    INTEGER, PARAMETER :: &
        xmin_crtm = 1, xmax_crtm = ix, ymin_crtm = 1, ymax_crtm = jx
    INTEGER, PARAMETER :: x_coarse = 1, y_coarse = 1  ! how many wrf gridpoints should be averaged together in a CRTM profile
    INTEGER, PARAMETER :: numx_crtm = 297 !floor((xmax_crtm - xmin_crtm + 1) / x_coarse) 
    INTEGER, PARAMETER :: numy_crtm = 297 !floor((ymax_crtm - ymin_crtm + 1) / y_coarse)            

    ! Variables for MPI and NETCDF
    integer, parameter :: NDIMS = 3 
    integer :: ncid,ncrcode,fid,x_dimid,y_dimid,ch_dimid,dimids(NDIMS),varid
    integer :: i, j, l
    integer :: grand_count, ystart, yend
    integer :: nyi
    ! Variables for subroutine
    integer :: x,y,z
    INTEGER :: ncl,icl,k1,k2  
    integer :: mp_scheme = 0  ! wsm6, gfdlfv3, thompson08
    logical :: l_use_default_re = .false. 
    
    ! Constants
    REAL, PARAMETER :: P1000MB=100000.D0
    REAL, PARAMETER :: R_D=287.D0
    REAL, PARAMETER :: Cpd=7.D0*R_D/2.D0
    REAL, PARAMETER :: Re=6378000.0
    REAL, PARAMETER :: MY_PI = 4*ATAN(1.)
    REAL, PARAMETER :: RADS_PER_DEGREE = MY_PI / 180.
    REAL, PARAMETER :: G=9.8

    ! Declare CRTM structures 
    TYPE(CRTM_ChannelInfo_type)             :: ChannelInfo(N_SENSORS)
    TYPE(CRTM_Geometry_type)                :: Geometry(N_PROFILES)
    TYPE(CRTM_Atmosphere_type)              :: Atm(N_PROFILES)
    TYPE(CRTM_Surface_type)                 :: Sfc(N_PROFILES)
    TYPE(CRTM_RTSolution_type), ALLOCATABLE :: RTSolution(:,:)
    TYPE(CRTM_Options_type)                 :: Options(N_PROFILES)

    ! ============================================================================
    ! **** PREPROCESS ****
    ! ============================================================================   
    call getarg(1,inputdir)
    call getarg(2,inputfile)
    call getarg(3,OUTPUT_DIR)
  
    ! Read resolution
    write(inputWRF,'(a,a)') trim(inputdir),trim(inputfile)
    write(*,*) trim(inputWRF)
    call open_file(inputWRF, nf_nowrite, fid)
    ncrcode = nf_get_att_real(fid, nf_global,'DX', dx)
    ncrcode = nf_get_att_real(fid, nf_global,'DY', dy)
    ! convert to km
    dx = dx / 1000.
    dy = dy / 1000.
    call close_file(fid)


    ! ============================================================================
    ! **** GET INPUT DATA ****
    ! ============================================================================

    ! Load Atmosphere and Surface input from WRF file
    ! --------------------------------
    call get_variable3d(inputWRF,'P',ix,jx,kx,1,p) ! perturbation pressure
    call get_variable3d(inputWRF,'PB',ix,jx,kx,1,pb) ! Base-state pressure 
    call get_variable3d(inputWRF,'PH',ix,jx,kx+1,1,ph) ! perturbation geopotential
    call get_variable3d(inputWRF,'PHB',ix,jx,kx+1,1,phb) ! Base-state geopotential 
    call get_variable3d(inputWRF,'T',ix,jx,kx,1,t)
    call get_variable3d(inputWRF,'QVAPOR',ix,jx,kx,1,qvapor)
    if(my_proc_id==0)  WRITE(*,*) 'use cloud: ', use_cloud
    if (use_cloud) then
        call get_variable3d(inputWRF,'QCLOUD',ix,jx,kx,1,qcloud)
        call get_variable3d(inputWRF,'QRAIN',ix,jx,kx,1,qrain)
        call get_variable3d(inputWRF,'QICE',ix,jx,kx,1,qice)
        call get_variable3d(inputWRF,'QSNOW',ix,jx,kx,1,qsnow)
        call get_variable3d(inputWRF,'QGRAUP',ix,jx,kx,1,qgraup)
    else
        qcloud = 0.
        qrain = 0.
        nrain = 0.
        qice = 0.
        qsnow = 0.
        qgraup = 0.
    endif
    call get_variable2d(inputWRF,'PSFC',ix,jx,1,psfc) ! Surface pressure in Pa
    call get_variable2d(inputWRF,'TSK',ix,jx,1,tsk) ! Surface skin temperature
    call get_variable2d(inputWRF,'XLAND',ix,jx,1,xland)
    call get_variable2d(inputWRF,'HGT',ix,jx,1,hgt) ! Terrian height
    call get_variable2d(inputWRF,'U10',ix,jx,1,u10) ! 10-m U wind
    call get_variable2d(inputWRF,'V10',ix,jx,1,v10) ! 10-m V wind
    call get_variable2d(inputWRF, 'XLONG', ix, jx, 1, xlong )
    call get_variable2d(inputWRF, 'XLAT', ix, jx, 1, xlat ) 
    ! Convert degrees to radians
    lat = xlat/180.0*MY_PI 
    lon = xlong/180.0*MY_PI
    ! Pressure
    pres = P + PB
    tk = (T + 300.0) * ( (pres / P1000MB) ** (R_D/Cpd) )
    !write(*,*) 'Pressures from WRF : ', pres(1,2,:)
    !write(*,*) 'Temperatures from WRF : ', tk(1,2,:)
    ! Physical constraints on mixing ratios (>=0)
    where(qvapor.lt.0.0) qvapor=1.0e-8
    where(qcloud.lt.0.0) qcloud=0.0
    where(qice.lt.0.0) qice=0.0
    where(qrain.lt.0.0) qrain=0.0
    where(nrain.lt.0.0) nrain=0.0
    where(qsnow.lt.0.0) qsnow=0.0
    where(qgraup.lt.0.0) qgraup=0.0
    ! Wind
    westwind = (U10 .lt. 0)
    ! Wind speed
    windspeed = sqrt(U10**2 + V10**2)
    where (ISNAN(windspeed)) windspeed = 0.0d0
    ! Wind direction
    ! corrected for CRTM version 2.3.0
    ! winddir = -180*(westwind) + ( 90 - ( atan(V10/U10)/RADS_PER_DEGREE) )
    winddir = 180*(1 - westwind) + (90 - ( atan(V10/U10)/RADS_PER_DEGREE) )
    where (ISNAN(winddir)) winddir = 0.0d0

    ! ============================================================================
    ! **** INITIALIZE CRTM AND ALLOCATE STRUCTURE ARRAYS ****
    ! ============================================================================
  
    ! This initializes the CRTM for the sensors
    ! predefined in the example SENSOR_ID parameter.
    ! Its purpose is to generate a structure/object that
    ! contains the satellite/sensor channel index information.
    ! NOTE: The coefficient data file path is hard-
    !       written for this example.
    ! --------------------------------------------------
    call parallel_start()
    call CRTM_Version( Version )

    ! If sensor class is either gpm_gmi_lf or gmi_gpm_hf (low- or high-frequency),
    ! then initialize the CRTM with 'gpm_gmi'. Otherwise, use the actual Sensor_ID
    ! specified by the observation.
    ! 'gpm_gmi_lf' is channels 1-9, while 'gpm_gmi_hg' is channels 10-13. 
    ! These two sets of channels have different scan and zenith angles. If observations
    ! from both of these two sets of channels are assimilated, then treating these sets of
    ! channels as different classes of sensors is necessary to have the calculated average 
    ! scan and zenith angles of observations in the vicinity of a given model grid point 
    ! to be correct for each observation. However, the CRTM still needs to be told 'gmi_gpm'
    ! for either.
    if(INDEX(trim(Sensor_Id), 'gmi')>0) then
      Sensor_Id = 'gmi_gpm'
    else
      Sensor_Id = trim(Sensor_Id)
    endif

    if(my_proc_id==0)  write(*,*)
    if(my_proc_id==0)  write(*,*) "CRTM ver.",TRIM(Version)
    if(my_proc_id==0) WRITE( *,'(/5x,"Initializing the CRTM...")' )
        Error_Status = CRTM_Init( (/Sensor_Id/), &  ! Input... must be an array, hencethe (/../)
                                ChannelInfo  , &  ! Output
                                IRwaterCoeff_File='WuSmith.IRwater.EmisCoeff.bin',&
                                IRlandCoeff_File='IGBP.IRland.EmisCoeff.bin',&
                                File_Path='coefficients/', &
                                CloudCoeff_File_rain  = 'Thompson08_RainLUT_-109z-1.bin',&
                                CloudCoeff_File_snow  = 'Thompson08_SnowLUT_-109z-1.bin',&
                                CloudCoeff_File_graup = 'Thompson08_GraupelLUT_-109z-1.bin',&
                                !CloudCoeff_File_rain  = 'WSM6_RainLUT_-109z-1.bin',&
                                !CloudCoeff_File_snow  = 'WSM6_SnowLUT_-109z-1.bin',&
                                !CloudCoeff_File_graup = 'WSM6_GraupelLUT_-109z-1.bin',&
                                Quiet=.false.)

    IF ( Error_Status /= SUCCESS ) THEN
      Message = 'Error initializing CRTM'
      CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
      STOP
    END IF 

    ! Only process a subset of channels
    n_channels = SUM(CRTM_ChannelInfo_n_Channels(ChannelInfo))
    if(my_proc_id==0)  write(*,*) "Number of channels initialized from CRTM_init",n_Channels

    Error_Status = CRTM_ChannelInfo_Subset( ChannelInfo(1), Channel_Subset = channel_subset )
    IF ( Error_Status /= SUCCESS ) THEN
        Message = 'Error making a channel subset'
        CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
        STOP
    END IF
    n_channels = size(channel_subset)
    
    
    ! Allocate RTSolution ARRAYS
    ALLOCATE( RTSolution( n_channels, N_PROFILES ), STAT=Allocate_Status )      
    IF ( Allocate_Status /= 0 ) THEN
      Message = 'Error allocating structure arrays'
      CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
      STOP
    END IF

    ! Allocate Atmosphere STRUCTURES
    CALL CRTM_Atmosphere_Create( Atm, kx, N_ABSORBERS, N_CLOUDS, N_AEROSOLS)
    IF ( ANY(.NOT. CRTM_Atmosphere_Associated(Atm)) ) THEN
      Message = 'Error allocating CRTM Atmosphere structures'
      CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
      STOP
    END IF 

    ! Allocate Tb arrays
    ALLOCATE(          Tb(numx_crtm,numy_crtm,n_channels), STAT = Allocate_Status)
    ALLOCATE(      Tbsend(numx_crtm,numy_crtm,n_channels), STAT = Allocate_Status)
    TB = 0.0d0     
    Tbsend = 0.0d0   

    ! Run CRTM on each grid point with help of MPI
    ! -------------------------------
    grand_count = 0
    ! nprocs: size of the communicator (how many processors inside)
    if(mod(numy_crtm,nprocs).eq.0) then
       nyi=numy_crtm/nprocs
    else
       nyi=numy_crtm/nprocs+1
    endif
    ! nyi: rows are divided into nyi slabs and each processor processes one.
    ystart=my_proc_id+1
    ! my_proc_id is from 0 to size-1; ystart: 1 to size.
    yend=numy_crtm    


    ! %%%%%%%%%%%%%%%%  LOOP: start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    loop_y: do j=ystart, yend, nprocs
            IF (SUPPRESS_OUTPUT) THEN
                WRITE(*,*) 'proc ', my_proc_id, 'is at wrf domain row', j+ymin_crtm-1
            END IF
    loop_x: do i=1, numx_crtm
            grand_count = grand_count + 1
            IF ( .NOT. SUPPRESS_OUTPUT ) THEN
                WRITE(*,*) 'proc ', my_proc_id, ' is at point x=', i+xmin_crtm-1, ' y=', j+ymin_crtm-1
            END IF     
            ! time (It's not needed to build training database for CCV operator)
            ! YEAR_STR = times(1:4)
            ! MONTH_STR = times(6:7)
            ! DAY_STR = times(9:10)
            ! read(YEAR_STR,*) YEAR
            ! read(MONTH_STR,*) MONTH
            ! read(DAY_STR,*) DAY
            
            CALL CRTM_Geometry_SetValue( Geometry, &
                                         Sensor_Zenith_Angle  = my_zenith_angle, &
                                         Sensor_Scan_Angle    = my_scan_angle)!, &
                                         !Sensor_Azimuth_Angle = my_azimuth_angle) !, &
                                         !Year                 = YEAR, &
                                         !Month                = MONTH, &
                                         !Day                  = DAY ) 
            ! If use slant-path method   
            ! --------------------------------                
            if (use_slant_path) then
              call Load_CRTM_Structures_MW_slant( Atm, Sfc, i, j, kx, &
                       my_zenith_angle, my_azimuth_angle, dx, .FALSE.)
            else
              call Load_CRTM_Structures_MW( Atm, Sfc, i, j, kx, .FALSE.)
            end if
              
            ! Use the SOI radiative transfer algorithm
            ! --------------------------------------------
            Options%RT_Algorithm_ID = RT_SOI
            
            ! Specify number of streams
            ! --------------------------------------------
            IF (N_STREAMS .GT. 0) THEN
              Options%Use_N_Streams = .TRUE.
              Options%n_Streams = N_STREAMS
            END IF

            ! ============================================================================
            ! **** CALL THE CRTM FORWARD MODEL ****
            ! ============================================================================
            Error_Status = CRTM_Forward( Atm        , &
                                         Sfc        , &
                                         Geometry   , &
                                         ChannelInfo, &
                                         RTSolution , & !n_sensors x n_profiles
                                         Options = Options )
            IF ( Error_Status /= SUCCESS ) THEN
              Message = 'Error in CRTM Forward Model'
              CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
              STOP
            END IF

            ! ============================================================================
            ! **** COLLECT OUTPUT OF A SINGLE POINT ****
            !
            ! User should read the user guide or the source code of the routine
            ! CRTM_RTSolution_Inspect in the file CRTM_RTSolution_Define.f90 to
            ! select the needed variables for outputs.  These variables are contained
            ! in the structure RTSolution.
            ! ============================================================================
            !---for file output, edited 2014.9.26
            do l = 1, n_channels
                Tbsend(i,j,l) = real(RTSolution(l,1)%Brightness_Temperature)
                if ((i .EQ. 121 .AND. j .EQ. 1)) then
                  WRITE(*,*) '  at x=121, y=1, Tbsend=',Tbsend(i,j,l)
                endif
                !if (i .EQ. 1 .AND. j .EQ. 2) then
                !    write(*,*) 'Pressures: ', atm(1)%Pressure
                !    write(*,*) 'Level_pressures: ', atm(1)%Level_Pressure
                !    write(*,*) 'Temperatures: ', atm(1)%Temperature
                !    write(*,*) 'Absorbers: ', atm(1)%Absorber
                !endif
                ! Tb might be NAN
                if (Tbsend(i,j,l) .NE. Tbsend(i,j,l)) then
                  write(*,*) '  Tbsend is NaN at x=',i,' y=',j
                  write(*,*) 'Pressures: ', atm(1)%Pressure
                  write(*,*) 'Level_pressures: ', atm(1)%Level_Pressure
                endif
                ! Tb might not make physical sense
                if (Tbsend(i,j,l) .GT. 999) then
                  WRITE(*,*) '  at x=',i,'y=',j,'Tbsend=',Tbsend(i,j,l)
                  write(*,*) 'Pressures: ', atm(1)%Pressure
                  write(*,*) 'Level_pressures: ', atm(1)%Level_Pressure
                endif
            enddo

    enddo loop_x ! loop over x dimension
    enddo loop_y ! loop over y dimension
    ! %%%%%%%%%%%%%%%% LOOP: end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    CALL MPI_Allreduce(Tbsend,Tb,numx_crtm*numy_crtm*n_channels,MPI_REAL,MPI_SUM,comm,ierr)
    WRITE(*,*) 'successfully called MPI_Allreduce for Tb'

    ! ============================================================================
    !  **** writing the output ****
    ! ============================================================================
    ! in netcdf format
    if(my_proc_id==0) then

        write(OUTPUT_NAME,'(a,a,a,a,a)') trim(OUTPUT_DIR),trim(inputfile),'_',trim(EXPERIMENT_NAME),'.nc'
        write(*,*) trim(OUTPUT_NAME)
        ncrcode = nf_create(OUTPUT_NAME, NF_CLOBBER, ncid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
    
        ncrcode = nf_def_dim(ncid, "Longitude", numx_crtm,  x_dimid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
        ncrcode = nf_def_dim(ncid, "Latitude",  numy_crtm,  y_dimid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
        ncrcode = nf_def_dim(ncid, "Channel",   n_channels, ch_dimid)
        if(ncrcode /= nf_noerr) then 
            print *, trim(nf_strerror(status))
        stop "Stopped"
        end if
        dimids =  (/ x_dimid, y_dimid, ch_dimid /)

        ncrcode = nf_def_var(ncid, "BT", NF_REAL, NDIMS, dimids, varid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
        ncrcode = nf_put_att_text(ncid, varid, "units", 6, "Kelvin")
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
      
        ncrcode = nf_enddef(ncid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
      
        ncrcode = nf_put_var_real(ncid, varid, Tb)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
        stop "Stopped"
        end if
        write(*,*) '8'

        ncrcode = nf_close(ncid)
        if(ncrcode /= nf_noerr) then
            print *, trim(nf_strerror(status))
            stop "Stopped"
        end if
        write(*,*) '9'
    
    end if

    ! ============================================================================
    !  **** initializing all Tb and Tbsend fields ****
    !
    !  initializing the Tbsend fields for Bcast
    Tbsend = 0.0d0
    Tb = 0.0d0
    CALL MPI_BCAST(Tbsend,numx_crtm*numy_crtm*n_channels,MPI_REAL,0,comm,ierr)



    ! ============================================================================
    ! **** DESTROY THE CRTM ****
    ! ============================================================================
    WRITE( *, '( /5x, "Destroying the CRTM..." )' )
    Error_Status = CRTM_Destroy( ChannelInfo )
    IF ( Error_Status /= SUCCESS ) THEN
        Message = 'Error destroying CRTM'
        CALL Display_Message( PROGRAM_NAME, Message, FAILURE )
        STOP
    END IF
    ! ============================================================================
    call parallel_finish()


CONTAINS
  
  INCLUDE 'Load_CRTM_Structures_MW.inc'
  INCLUDE 'Load_CRTM_Structures_MW_slant.inc'

END PROGRAM crtm
