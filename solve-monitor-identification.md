# Solve problem of monitor identification in updating NVidia driver

In updating/installing new NVidia, your xorg.conf, Xorg server configuration file for displays and devices will look like something like this:

```vim
# nvidia-xconfig:  version 440.36


Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" 0 0
    InputDevice    "Keyboard0" "CoreKeyboard"
    InputDevice    "Mouse0" "CorePointer"
EndSection

Section "Files"
EndSection

Section "InputDevice"

    # generated from default
    Identifier     "Mouse0"
    Driver         "mouse"
    Option         "Protocol" "auto"
    Option         "Device" "/dev/psaux"
    Option         "Emulate3Buttons" "no"
    Option         "ZAxisMapping" "4 5"
EndSection

Section "InputDevice"

    # generated from default
    Identifier     "Keyboard0"
    Driver         "kbd"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "Unknown"
    ModelName      "Unknown"
    Option         "DPMS"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Monitor        "Monitor0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Modes      "1920x1080" "1920x1080"           
    EndSubSection
EndSection
```

If one of your monitors are blank, it means that your device cant work/detect the both displays. You can confirm this by the command ```bash xrandr -q```, which shows all detected components.
You will see something like this:

```bash
Screen 0: minimum 8 x 8, current 3840 x 1080, maximum 32767 x 32767                                                                                                                                                
HDMI-0 connected 1920x1080+1920+0 (normal left inverted right x axis y axis) 480mm x 270mm                                                                                                                         
   1920x1080     60.00 +  59.94*   50.00    60.00    50.04                                                                                                                                                         
   1680x1050     59.95                                                                                                                                                                                             
   1440x900      59.89                                                                                                                                                                                             
   1280x1024     75.02    60.02                                                                                                                                                                                    
   1280x960      60.00                                                                                                                                                                                             
   1280x720      60.00    59.94    50.00                                                                                                                                                                           
   1024x768      75.03    70.07    60.00                                                                                                                                                                           
   800x600       75.00    72.19    60.32    56.25                                                                                                                                                                  
   720x576       50.00                                                                                                                                                                                             
   720x480       59.94                                                                                                                                                                                             
   640x480       75.00    72.81    59.94    59.93
```

The following NVIDIA forum post confirms that you cant handle all displays in NVIDIA unless you use Prime.

https://devtalk.nvidia.com/default/topic/1062629/ubuntu-18-04-single-xorg-screen-config-with-internal-intel-graphics-and-nvidia-egpu/?offset=4

Whatever Prime is, there is an easier solution, if you gonna use your GPU only for intensive data algorithms processing.

Use ```bash $lspci``` to list all your pci devices. Look for VGA controller:

```bash
$lspci | grep VGA
00:02.0 VGA compatible controller: Intel Corporation Device 3e9b
01:00.0 VGA compatible controller: NVIDIA Corporation Device 1c91 (rev a1)
``` 

Use the same command again, without grep, and search for the driver of the non NVIDIA device, in my case the driver was i915;

Change your ```bash /etc/X11/xorg.conf``` file to something like this:

```vim
# nvidia-xconfig:  version 440.36


Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0" 0 0
    InputDevice    "Keyboard0" "CoreKeyboard"
    InputDevice    "Mouse0" "CorePointer"
EndSection

Section "Files"
EndSection

Section "InputDevice"

    # generated from default
    Identifier     "Mouse0"
    Driver         "mouse"
    Option         "Protocol" "auto"
    Option         "Device" "/dev/psaux"
    Option         "Emulate3Buttons" "no"
    Option         "ZAxisMapping" "4 5"
EndSection

Section "InputDevice"

    # generated from default
    Identifier     "Keyboard0"
    Driver         "kbd"
EndSection

Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "Unknown"
    ModelName      "Unknown"
    Option         "DPMS"
EndSection

Section "Device"
    Identifier     "Device0"
    Driver         "i915"
    VendorName     "Intel Corporation"
EndSection

Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    Monitor        "Monitor0"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Modes      "1920x1080" "1920x1080"           
    EndSubSection
EndSection
```

Reboot. You probably got your dual monitor setup again. But will be unable to use it for displaying ... (no games maybe ? i dont give a damn, but many people wont like lol...).

To confirm the detection of your monitors, issue ```bash xrandr -q``` again.

You will see something like this:

```bash $xrandr -q

Screen 0: minimum 8 x 8, current 3840 x 1080, maximum 32767 x 32767
HDMI-0 connected 1920x1080+1920+0 (normal left inverted right x axis y axis) 480mm x 270mm
   1920x1080     60.00 +  59.94*   50.00    60.00    50.04
   1680x1050     59.95
   1440x900      59.89
   1280x1024     75.02    60.02
   1280x960      60.00
   1280x720      60.00    59.94    50.00
   1024x768      75.03    70.07    60.00
   800x600       75.00    72.19    60.32    56.25
   720x576       50.00
   720x480       59.94
   640x480       75.00    72.81    59.94    59.93
eDP-1-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis) 344mm x 194mm
   1920x1080     59.98*+  59.97    59.96    59.93    47.98
   1680x1050     59.95    59.88
   1600x1024     60.17
   1400x1050     59.98
   1600x900      59.99    59.94    59.95    59.82
   1280x1024     60.02
   1440x900      59.89
   1400x900      59.96    59.88
   1280x960      60.00
   1440x810      60.00    59.97
   1368x768      59.88    59.85
   1360x768      59.80    59.96
   1280x800      59.99    59.97    59.81    59.91
   1152x864      60.00
   1280x720      60.00    59.99    59.86    59.74
   1024x768      60.04    60.00
   960x720       60.00
   928x696       60.05
   896x672       60.01
   1024x576      59.95    59.96    59.90    59.82
   960x600       59.93    60.00
   960x540       59.96    59.99    59.63    59.82
   800x600       60.00    60.32    56.25
   840x525       60.01    59.88
   864x486       59.92    59.57
   800x512       60.17
   700x525       59.98
   800x450       59.95    59.82
   640x512       60.02
   720x450       59.89
   700x450       59.96    59.88
   640x480       60.00    59.94
   720x405       59.51    58.99
   684x384       59.88    59.85
   680x384       59.80    59.96
   640x400       59.88    59.98
   576x432       60.06
   640x360       59.86    59.83    59.84    59.32
   512x384       60.00
   512x288       60.00    59.92
   480x270       59.63    59.82
   400x300       60.32    56.34
   432x243       59.92    59.57
   320x240       60.05
   360x202       59.51    59.13
   320x180       59.84    59.32
DP-1-1 disconnected (normal left inverted right x axis y axis)
HDMI-1-1 disconnected (normal left inverted right x axis y axis)
  1680x1050 (0x4b) 146.250MHz -HSync +VSync
        h: width  1680 start 1784 end 1960 total 2240 skew    0 clock  65.29KHz
        v: height 1050 start 1053 end 1059 total 1089           clock  59.95Hz
  1280x1024 (0x53) 108.000MHz +HSync +VSync
        h: width  1280 start 1328 end 1440 total 1688 skew    0 clock  63.98KHz
        v: height 1024 start 1025 end 1028 total 1066           clock  60.02Hz
  1440x900 (0x54) 106.500MHz -HSync +VSync
        h: width  1440 start 1520 end 1672 total 1904 skew    0 clock  55.93KHz
        v: height  900 start  903 end  909 total  934           clock  59.89Hz
  1280x960 (0x57) 108.000MHz +HSync +VSync
        h: width  1280 start 1376 end 1488 total 1800 skew    0 clock  60.00KHz
        v: height  960 start  961 end  964 total 1000           clock  60.00Hz
  1024x768 (0x68) 65.000MHz -HSync -VSync
        h: width  1024 start 1048 end 1184 total 1344 skew    0 clock  48.36KHz
        v: height  768 start  771 end  777 total  806           clock  60.00Hz
  800x600 (0x77) 40.000MHz +HSync +VSync
        h: width   800 start  840 end  968 total 1056 skew    0 clock  37.88KHz
        v: height  600 start  601 end  605 total  628           clock  60.32Hz
  800x600 (0x78) 36.000MHz +HSync +VSync
        h: width   800 start  824 end  896 total 1024 skew    0 clock  35.16KHz
        v: height  600 start  601 end  603 total  625           clock  56.25Hz
  640x480 (0x86) 25.175MHz -HSync -VSync
        h: width   640 start  656 end  752 total  800 skew    0 clock  31.47KHz
        v: height  480 start  490 end  492 total  525           clock  59.94Hz
```

The eDP-1-1 thing must be my notebook internal monitor. The disconnected things are artefacts which come probably, but i wont lose time looking at it (too lazy for configuring boring things), due to my unnecessary SubSection in Section "Screen" of xorg.conf file.

So... Happy Deving with dual monitors...

