#!/bin/bash
# Cleanup script to organize the repository for release

mkdir -p _legacy_data

echo "Moving legacy files to _legacy_data/..."
mv paper_figure.ipynb _legacy_data/ 2>/dev/null
mv benchmark_results.txt _legacy_data/ 2>/dev/null
mv *.nc _legacy_data/ 2>/dev/null

echo "Repository cleaned."
echo "Ready files:"
ls -1
