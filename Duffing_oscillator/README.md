# Duffing oscillator

## Differential equation
![equations](./equations_duffing.png)

## Poincaré section
![example](./poincare.gif)

The poincare section is progressively modified during the animation.

## Commands to make gif from images
- png to mp4 to gif
  - ```ffmpeg -framerate 30 -i frame_%05d.png -vcodec libx264 -crf 15 -r 30 -pix_fmt yuv420p out.mp4```
  - ```ffmpeg -ss 0.0 -i out.mp4 -f gif out.gif```
- png to gif using a palette
  - ```ffmpeg -i frame_%05d.png -vf palettegen=reserve_transparent=1 palette.png```
  - ```ffmpeg -framerate 30 -i frame_%05d.png -i palette.png -lavfi paletteuse=alpha_threshold=128 -gifflags -offsetting out.gif```