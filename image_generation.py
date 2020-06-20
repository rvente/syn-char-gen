#!/usr/bin/python3
# Ralph "Blake" Vente
# Takes in an alphabet and generates synthetic examples

from cv2 import cv2
import numpy as np
# docs: https://jeltef.github.io/PyLaTeX/current/examples/environment_ex.html
# from pylatexenc.latex2text import LatexNodes2Text
from pylatex import Document, NoEscape, Package, Command
from pylatex.utils import bold, italic

from icecream import ic
from tqdm import tqdm

import subprocess
from glob import glob
import os
from pathlib import Path

matcher     = r"{{template}}"
boilerplate = r"""\documentclass{minimal} \begin{document} {{template}} \end{document}"""
boilerplate = matcher

SLASH = "\\"

def surround(child, parent = "", prefix=SLASH):
  return prefix+parent+"{"+child+"}"

# TODO: Refactor this
def alphabet():
  for i in range(ord('a'), ord('z')+1): yield chr(i)
  for j in range(ord('A'), ord('Z')+1): yield chr(j)
  # for k in range(ord('α'), ord('ω')+1): yield chr(k)
  # for l in range(ord('Α'), ord('Ρ')+1): yield chr(l)
  # for m in range(ord('Σ'), ord('Ω')+1): yield chr(m)

# permute character combination
def char_combos():
  plain = lambda x: x
  cmds  = [plain, bold, italic]
  for cmd in cmds:
    for c in alphabet():
      yield cmd(c)

def main():
  geometry_options = {
    "margin": "0in",
    "headheight": "0pt",
    "headsep": "0pt",
    "voffset": "0mm", # no change
    "hoffset": "1mm",
    "includeheadfoot": False,
    "paperheight": "5mm",
    "textheight":"5mm",
    "textwidth":"5mm",
    "paperwidth": "7mm",
    "marginparwidth":"0mm"
  }
  for char_combo in char_combos():
    doc = Document(
      geometry_options=geometry_options,
      documentclass="article",
      # document_options="convert",
      indent=False
    )
    doc.append(char_combo)
    file_path = "./pdf/"+char_combo.replace("\\","").replace("{", "").replace("}","")
    doc.generate_pdf(file_path, compiler="lualatex", compiler_args=["-shell-escape"])

def convert():
  for token in alphabet():
    command = f"convert -density 500 {token}.pdf -background white -alpha remove -alpha off ../png/{token}.png".split()
    try:
      _ = subprocess.call(command, cwd="./pdf")
    except Exception as e:
      print(e)
      continue

def center_image(image):
  height, width = image.shape
  wi=(width/2)
  he=(height/2)

  ret,thresh = cv2.threshold(image,95,255,0)

  M = cv2.moments(thresh)

  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])

  offsetX = (wi-cX)
  offsetY = (he-cY)
  T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
  centered_image = cv2.warpAffine(image, T, (width, height))

  return centered_image

def apply_center(unctr_img_dir):
  """ Optically center a character in a square frame
  """
  uncentered_image_names = glob(unctr_img_dir)
  DIR = Path(".")
  for img_name in uncentered_image_names:

    # normalize images
    img = ~cv2.imread(img_name, cv2.COLOR_BGR2GRAY)
    IMG = os.path.basename(img_name)

    h, w = img.shape
    # h = w = max(w,h) 
    h = int(h//2 )
    w = int(w//2 )

    dst = center_image(img)
    hmargin = 30
    vmargin = 10
    dst = dst[vmargin:-vmargin, hmargin:-hmargin]
    outputPath = str((DIR/"png_normalized"/IMG))
    cv2.imwrite(outputPath, ~dst)



if __name__ == "__main__":
  # main()
  # convert()
  apply_center("./png/*")


