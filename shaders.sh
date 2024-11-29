#!/bin/bash

# Directory containing the shaders
SHADER_DIR="shaders"

# Create an output directory for SPIR-V files
OUTPUT_DIR="${SHADER_DIR}"

# Compile all .vert, .frag, .comp, .geom, and .tesc/tese files
for shader in "$SHADER_DIR"/*.{glsl,vert,frag,comp,geom,tesc,tese}; do
  # Check if the file exists (handles globbing when no files match)
  if [[ -f "$shader" ]]; then
    # Extract the file name and extension
    filename=$(basename -- "$shader")
    extension="${filename##*.}"
    name="${filename%.*}"

    # Output SPIR-V file path
    output_file="${OUTPUT_DIR}/${name}.${extension}.spv"

    echo "Compiling $shader -> $output_file"

    # Compile the shader
    glslangValidator -V "$shader" -o "$output_file"
    if [[ $? -ne 0 ]]; then
      echo "Error: Failed to compile $shader"
    else
      echo "Compiled successfully!"
    fi
  fi
done

echo "All shaders processed. SPIR-V files are in $OUTPUT_DIR"

