document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const uploadInput = document.getElementById('image-upload');
    const canvas = document.getElementById('blot-canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const emptyState = document.getElementById('empty-state');
    const clearBtn = document.getElementById('clear-rois');
    const exportBtn = document.getElementById('export-csv');
    const invertToggle = document.getElementById('invert-toggle');
    const tbody = document.getElementById('results-body');
    const bgBadge = document.getElementById('bg-badge');

    // Radio button handling for custom cards
    const radioCards = document.querySelectorAll('.radio-card');
    let currentRoiMode = 'target';

    radioCards.forEach(card => {
        card.addEventListener('click', () => {
            radioCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            const radio = card.querySelector('input[type="radio"]');
            radio.checked = true;
            currentRoiMode = radio.value;
        });
    });

    // State Variables
    let imageObj = null;
    let baseImageData = null; // Original image data
    let drawnROIs = []; // Array of ROI objects
    let isDrawing = false;
    let startX, startY;
    let currentRect = null;
    let renderScale = 1;

    // ROI Colors
    const colors = {
        target: '#3b82f6',
        control: '#10b981',
        background: '#ef4444'
    };

    // Load Image
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                imageObj = img;
                setupCanvas();
                emptyState.style.display = 'none';
                canvas.style.display = 'block';
                drawnROIs = []; // reset ROIs
                updateResults();
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    function setupCanvas() {
        if (!imageObj) return;

        // Container dimensions
        const wrapper = document.getElementById('canvas-wrapper');
        const padding = 32;
        const maxWidth = wrapper.clientWidth - padding;
        const maxHeight = wrapper.clientHeight - padding;
        
        // Calculate scale to fit
        const scaleX = maxWidth / imageObj.width;
        const scaleY = maxHeight / imageObj.height;
        renderScale = Math.min(1, scaleX, scaleY); // Don't scale up, only down

        // Set actual canvas resolution to original image
        canvas.width = imageObj.width;
        canvas.height = imageObj.height;
        
        // Use CSS to scale it visually
        canvas.style.width = `${imageObj.width * renderScale}px`;
        canvas.style.height = `${imageObj.height * renderScale}px`;

        redrawCanvas();

        // Cache original image data for calculations
        ctx.drawImage(imageObj, 0, 0);
        baseImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    }

    // Handle Window Resize
    window.addEventListener('resize', () => {
        if (imageObj) setupCanvas();
    });

    // Toggle Invert (Affects visual but densitometry always calculates assuming bright band on dark bg if invert is checked)
    invertToggle.addEventListener('change', () => {
        redrawCanvas();
        updateResults(); // Re-calculate everything
    });

    // Drawing Logic
    canvas.addEventListener('mousedown', (e) => {
        if (!imageObj) return;
        const rect = canvas.getBoundingClientRect();
        startX = (e.clientX - rect.left) / renderScale;
        startY = (e.clientY - rect.top) / renderScale;
        isDrawing = true;
        
        currentRect = { x: startX, y: startY, w: 0, h: 0, type: currentRoiMode };
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const currentX = (e.clientX - rect.left) / renderScale;
        const currentY = (e.clientY - rect.top) / renderScale;
        
        currentRect.w = currentX - startX;
        currentRect.h = currentY - startY;
        
        redrawCanvas();
        drawRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h, colors[currentRect.type]);
    });

    canvas.addEventListener('mouseup', () => {
        if (!isDrawing) return;
        isDrawing = false;
        
        // Ensure w and h are positive
        if (currentRect.w < 0) {
            currentRect.x += currentRect.w;
            currentRect.w = Math.abs(currentRect.w);
        }
        if (currentRect.h < 0) {
            currentRect.y += currentRect.h;
            currentRect.h = Math.abs(currentRect.h);
        }

        // Only add if it's a drag, not just a click
        if (currentRect.w > 5 && currentRect.h > 5) {
            currentRect.id = Date.now().toString().slice(-5);
            drawnROIs.push({...currentRect});
            
            // Auto-switch back to target if we just drew a background
            if(currentRoiMode === 'background' || currentRoiMode === 'control') {
                 document.getElementById('roi-target').click();
            }

            updateResults();
        }
        redrawCanvas();
        currentRect = null;
    });

    function drawRect(x, y, w, h, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 / renderScale; // Keep line width visually consistent
        ctx.strokeRect(x, y, w, h);
        
        // Semi-transparent fill
        ctx.fillStyle = `${color}33`; // 20% opacity hex
        ctx.fillRect(x, y, w, h);
    }

    function redrawCanvas() {
        if (!imageObj) return;
        
        // Draw base image
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (invertToggle.checked && baseImageData) {
            // Draw inverted
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            const tCtx = tempCanvas.getContext('2d');
            const imgData = new ImageData(
                new Uint8ClampedArray(baseImageData.data),
                canvas.width,
                canvas.height
            );
            for (let i = 0; i < imgData.data.length; i += 4) {
                imgData.data[i] = 255 - imgData.data[i];     // R
                imgData.data[i+1] = 255 - imgData.data[i+1]; // G
                imgData.data[i+2] = 255 - imgData.data[i+2]; // B
                // Alpha remains same
            }
            tCtx.putImageData(imgData, 0, 0);
            ctx.drawImage(tempCanvas, 0, 0);
        } else {
            ctx.drawImage(imageObj, 0, 0);
        }

        // Draw saved ROIs
        drawnROIs.forEach((roi, index) => {
            drawRect(roi.x, roi.y, roi.w, roi.h, colors[roi.type]);
            // Draw Label
            ctx.fillStyle = colors[roi.type];
            const fontSize = Math.max(12, 14/renderScale);
            ctx.font = `${fontSize}px Inter`;
            // Keep label inside bounds somewhat
            ctx.fillText(`${roi.type.charAt(0).toUpperCase()}${index+1}`, roi.x + 5/renderScale, roi.y + fontSize + 2/renderScale);
        });
    }

    // Quantitative Calculations
    function calculateROIStats(roi, imgData) {
        let sum = 0;
        let count = 0;
        
        const xStart = Math.min(Math.max(Math.floor(roi.x), 0), canvas.width);
        const yStart = Math.min(Math.max(Math.floor(roi.y), 0), canvas.height);
        const xEnd = Math.min(Math.ceil(roi.x + roi.w), canvas.width);
        const yEnd = Math.min(Math.ceil(roi.y + roi.h), canvas.height);
        
        const invertIntensities = invertToggle.checked;

        for (let y = yStart; y < yEnd; y++) {
            for (let x = xStart; x < xEnd; x++) {
                const idx = (y * canvas.width + x) * 4;
                // Standard Grayscale conversion (Luma)
                let r = imgData.data[idx];
                let g = imgData.data[idx+1];
                let b = imgData.data[idx+2];
                let grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
                
                if (invertIntensities) {
                    // Densitometry assumes bright bands on dark background
                    // If UI invert is checked, the original image is dark bands on light bg
                    // So we invert the intensity calculation here.
                    grayscale = 255 - grayscale;
                }
                
                sum += grayscale;
                count++;
            }
        }
        
        return {
            area: count,
            meanInt: count > 0 ? (sum / count) : 0,
            rawExt: sum // Area * MeanInt
        };
    }

    function updateResults() {
        if (!imageObj) return;

        // Make sure we have the pure original image data for calculations
        // (Visuals might be inverted, but we always want to calculate accurately)
        const imgData = baseImageData;

        // 1. Calculate base stats for all ROIs
        drawnROIs.forEach(roi => {
            const stats = calculateROIStats(roi, imgData);
            roi.area = stats.area;
            roi.meanInt = stats.meanInt;
            roi.rawExt = stats.rawExt;
        });

        // 2. Background Calculation
        const bgRois = drawnROIs.filter(r => r.type === 'background');
        let globalBgMean = 0;
        if (bgRois.length > 0) {
            let totalBgSum = bgRois.reduce((acc, r) => acc + (r.meanInt * r.area), 0);
            let totalBgArea = bgRois.reduce((acc, r) => acc + r.area, 0);
            globalBgMean = totalBgArea > 0 ? (totalBgSum / totalBgArea) : 0;
            bgBadge.textContent = 'Background Subtracted';
            bgBadge.style.color = 'var(--success)';
            bgBadge.style.background = 'rgba(16, 185, 129, 0.1)';
            bgBadge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
        } else if (drawnROIs.length > 0) {
            bgBadge.textContent = 'No Background Set';
            bgBadge.style.color = 'var(--danger)';
            bgBadge.style.background = 'rgba(239, 68, 68, 0.1)';
            bgBadge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
        } else {
             bgBadge.textContent = 'Pending';
             bgBadge.style.color = 'var(--text-secondary)';
             bgBadge.style.background = 'transparent';
             bgBadge.style.borderColor = 'var(--panel-border)';
        }

        // 3. Apply Background Subtraction
        drawnROIs.forEach(roi => {
            if (roi.type === 'background') {
                roi.correctedMean = 0;
                roi.intDens = 0;
            } else {
                roi.correctedMean = Math.max(0, roi.meanInt - globalBgMean);
                roi.intDens = roi.correctedMean * roi.area;
            }
        });

        // 4. Relative Expresion vs. Loading Control
        // Assuming user might just have one control band for all, or multiple. Let's take the average of controls as baseline 1.0.
        // Actually, typically you compare sample A / control A. For simplicity here, we establish a global "Control Reference" from 1 or more control bands.
        const controlRois = drawnROIs.filter(r => r.type === 'control');
        let controlReference = 1;
        if (controlRois.length > 0) {
            let totalControlDensity = controlRois.reduce((acc, r) => acc + r.intDens, 0);
            controlReference = totalControlDensity / controlRois.length;
            if(controlReference === 0) controlReference = 1; // Prevent div zero
        }

        drawnROIs.forEach(roi => {
            if (roi.type === 'background') {
                roi.relDens = '-';
            } else if (roi.type === 'control') {
                roi.relDens = (roi.intDens / controlReference).toFixed(3);
            } else {
                roi.relDens = controlRois.length > 0 ? (roi.intDens / controlReference).toFixed(3) : '-';
            }
        });

        renderTable();
    }

    function renderTable() {
        tbody.innerHTML = '';
        if (drawnROIs.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="7">Draw regions of interest on the image to see data here.</td></tr>';
            exportBtn.disabled = true;
            return;
        }

        exportBtn.disabled = false;
        
        drawnROIs.forEach((roi, index) => {
            const tr = document.createElement('tr');
            
            const num = index + 1;
            const prefix = roi.type === 'target' ? 'T' : (roi.type === 'control' ? 'C' : 'BG');
            
            // Format numbers nicely
            const areaStr = Math.round(roi.area).toLocaleString();
            const meanStr = roi.meanInt.toFixed(2);
            const intDensStr = roi.type==='background' ? '-' : Math.round(roi.intDens).toLocaleString();

            tr.innerHTML = `
                <td><b>#${num}</b></td>
                <td><span class="type-indicator type-${roi.type}">${prefix}</span></td>
                <td>${areaStr}</td>
                <td>${meanStr}</td>
                <td><strong>${intDensStr}</strong></td>
                <td>${roi.relDens}</td>
                <td>
                    <button class="action-btn" data-id="${roi.id}" title="Remove ROI">
                        <i data-lucide="x"></i>
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Re-init lucide icons for new buttons
        lucide.createIcons();

        // Add delete listeners
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idToRemove = e.currentTarget.getAttribute('data-id');
                drawnROIs = drawnROIs.filter(r => r.id !== idToRemove);
                updateResults();
                redrawCanvas();
            });
        });
    }

    // Clear
    clearBtn.addEventListener('click', () => {
        drawnROIs = [];
        updateResults();
        redrawCanvas();
    });

    // Export CSV
    exportBtn.addEventListener('click', () => {
        if (drawnROIs.length === 0) return;

        let csvContent = "data:text/csv;charset=utf-8,";
        let headers = ["ROI_Number", "Type", "X", "Y", "Width", "Height", "Area_px", "Mean_Intensity", "Corrected_Mean", "Integrated_Density", "Relative_Density"];
        csvContent += headers.join(",") + "\r\n";

        drawnROIs.forEach((r, index) => {
            let row = [
                index + 1,
                r.type,
                Math.round(r.x),
                Math.round(r.y),
                Math.round(r.w),
                Math.round(r.h),
                r.area,
                r.meanInt.toFixed(4),
                r.correctedMean ? r.correctedMean.toFixed(4) : "0",
                r.intDens ? r.intDens.toFixed(4) : "0",
                r.relDens
            ];
            csvContent += row.join(",") + "\r\n";
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `western_blot_quantification_${new Date().toISOString().slice(0,10)}.csv`);
        document.body.appendChild(link); // Required for FF
        link.click();
        document.body.removeChild(link);
    });

});
