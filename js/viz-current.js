import React, { useState, useEffect } from 'react';

const TileOrderingVisualization = () => {
  // Default parameters from the original function
  const [patchSize, setPatchSize] = useState(512);
  const [vaeScaleFactor, setVaeScaleFactor] = useState(8);
  const [imageSize, setImageSize] = useState(2048);
  const [tileOrder, setTileOrder] = useState("corner_to_center");
  const [overlapRatio, setOverlapRatio] = useState(4);
  const [animationSpeed, setAnimationSpeed] = useState(300);
  const [animationRunning, setAnimationRunning] = useState(false);
  const [currentTileIndex, setCurrentTileIndex] = useState(-1);
  const [tiles, setTiles] = useState([]);
  
  // Calculated values
  const latentSize = imageSize / vaeScaleFactor;
  const overlapSizeLatent = (patchSize / vaeScaleFactor) / overlapRatio;
  const latentPatchSize = patchSize / vaeScaleFactor;
  
  const getOrderedTileCoordinates = () => {
    // Calculate effective patch size in latent space with overlap
    const effectivePatchSize = latentPatchSize - overlapSizeLatent;
    
    // Calculate how many tiles we need in each dimension
    const numTiles = Math.max(1, Math.floor((latentSize - overlapSizeLatent) / effectivePatchSize));
    
    // Create grid of tile centers
    let coordinates = [];
    
    // Add all grid coordinates
    for (let i = 0; i < numTiles; i++) {
      for (let j = 0; j < numTiles; j++) {
        // Calculate center of the tile in latent space
        const centerI = overlapSizeLatent + i * effectivePatchSize;
        const centerJ = overlapSizeLatent + j * effectivePatchSize;
        
        // Ensure the center is valid and not too close to the edge
        if (centerI >= overlapSizeLatent && centerI < latentSize - overlapSizeLatent &&
            centerJ >= overlapSizeLatent && centerJ < latentSize - overlapSizeLatent) {
          coordinates.push({
            x: centerI, 
            y: centerJ,
            type: 'grid'
          });
        }
      }
    }
    
    // Add edge and corner coordinates
    const halfPatch = latentPatchSize / 2;
    
    // Right edge centers
    const rightEdge = latentSize - halfPatch;
    for (let i = 0; i < numTiles; i++) {
      const centerI = overlapSizeLatent + i * effectivePatchSize;
      if (centerI >= overlapSizeLatent && centerI < latentSize - overlapSizeLatent) {
        coordinates.push({
          x: centerI, 
          y: rightEdge,
          type: 'edge'
        });
      }
    }
    
    // Bottom edge centers
    const bottomEdge = latentSize - halfPatch;
    for (let j = 0; j < numTiles; j++) {
      const centerJ = overlapSizeLatent + j * effectivePatchSize;
      if (centerJ >= overlapSizeLatent && centerJ < latentSize - overlapSizeLatent) {
        coordinates.push({
          x: bottomEdge, 
          y: centerJ,
          type: 'edge'
        });
      }
    }
    
    // Bottom right corner
    coordinates.push({
      x: bottomEdge, 
      y: rightEdge,
      type: 'corner'
    });
    
    // Order the coordinates based on the chosen ordering
    const center = latentSize / 2;
    if (tileOrder === "corner_to_center") {
      // Sort by distance from center (furthest first)
      coordinates.sort((a, b) => {
        const distA = Math.pow(a.x - center, 2) + Math.pow(a.y - center, 2);
        const distB = Math.pow(b.x - center, 2) + Math.pow(b.y - center, 2);
        return distB - distA;
      });
    } else { // center_to_corner
      // Sort by distance to center (closest first)
      coordinates.sort((a, b) => {
        const distA = Math.pow(a.x - center, 2) + Math.pow(a.y - center, 2);
        const distB = Math.pow(b.x - center, 2) + Math.pow(b.y - center, 2);
        return distA - distB;
      });
    }
    
    return coordinates;
  };
  
  useEffect(() => {
    setTiles(getOrderedTileCoordinates());
    setCurrentTileIndex(-1);
  }, [patchSize, vaeScaleFactor, imageSize, tileOrder, overlapRatio]);
  
  useEffect(() => {
    let timer;
    if (animationRunning && currentTileIndex < tiles.length - 1) {
      timer = setTimeout(() => {
        setCurrentTileIndex(prev => prev + 1);
      }, animationSpeed);
    } else if (currentTileIndex >= tiles.length - 1) {
      setAnimationRunning(false);
    }
    
    return () => clearTimeout(timer);
  }, [animationRunning, currentTileIndex, tiles.length, animationSpeed]);
  
  const startAnimation = () => {
    setCurrentTileIndex(-1);
    setAnimationRunning(true);
  };
  
  const pauseAnimation = () => {
    setAnimationRunning(false);
  };
  
  const resetAnimation = () => {
    setAnimationRunning(false);
    setCurrentTileIndex(-1);
  };
  
  const handleSliderChange = (index) => {
    setAnimationRunning(false);
    setCurrentTileIndex(index);
  };
  
  // Convert latent space coordinates to canvas coordinates
  const scale = 400 / latentSize;
  
  return (
    <div className="p-4 bg-white rounded-lg shadow-lg w-full max-w-4xl">
      <h2 className="text-2xl font-bold mb-4">Tile Ordering Visualization</h2>
      
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="lg:w-1/2">
          <div className="relative bg-gray-100 border border-gray-300" style={{ width: 400, height: 400 }}>
            {/* Draw latent space boundary */}
            <div className="absolute inset-0 border-2 border-gray-400"></div>
            
            {/* Draw all tiles that have been processed so far */}
            {tiles.slice(0, currentTileIndex + 1).map((tile, index) => {
              const centerX = tile.x * scale;
              const centerY = tile.y * scale;
              const halfPatchSize = (latentPatchSize / 2) * scale;
              
              // Different color based on tile type
              let bgColor = "bg-blue-500";
              if (tile.type === 'edge') bgColor = "bg-green-500";
              if (tile.type === 'corner') bgColor = "bg-red-500";
              
              return (
                <div key={index} 
                  className={`absolute ${bgColor} bg-opacity-30 border-2 border-opacity-60 flex items-center justify-center text-xs font-bold`} 
                  style={{
                    left: centerX - halfPatchSize,
                    top: centerY - halfPatchSize,
                    width: latentPatchSize * scale,
                    height: latentPatchSize * scale,
                    borderColor: index === currentTileIndex ? 'red' : 'rgba(0,0,0,0.5)',
                    zIndex: index === currentTileIndex ? 100 : 10 + index,
                  }}>
                  {index + 1}
                </div>
              );
            })}
          </div>
          
          <div className="flex gap-2 mt-4">
            <button 
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={startAnimation}
              disabled={animationRunning}>
              Play
            </button>
            <button 
              className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
              onClick={pauseAnimation}
              disabled={!animationRunning}>
              Pause
            </button>
            <button 
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              onClick={resetAnimation}>
              Reset
            </button>
          </div>
          
          <div className="mt-4">
            <label className="block text-sm font-medium mb-1">Tile Progress: {currentTileIndex + 1} of {tiles.length}</label>
            <input 
              type="range" 
              min="-1" 
              max={tiles.length - 1} 
              value={currentTileIndex}
              onChange={(e) => handleSliderChange(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
          
          <div className="mt-4">
            <label className="block text-sm font-medium mb-1">Animation Speed (ms): {animationSpeed}</label>
            <input 
              type="range" 
              min="50" 
              max="1000" 
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
        </div>
        
        <div className="lg:w-1/2">
          <h3 className="text-lg font-semibold mb-2">Parameters</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Patch Size: {patchSize}</label>
            <input 
              type="range" 
              min="128" 
              max="1024" 
              step="64"
              value={patchSize}
              onChange={(e) => setPatchSize(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Image Size: {imageSize}</label>
            <input 
              type="range" 
              min="1024" 
              max="4096" 
              step="256"
              value={imageSize}
              onChange={(e) => setImageSize(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">VAE Scale Factor: {vaeScaleFactor}</label>
            <input 
              type="range" 
              min="4" 
              max="16" 
              step="1"
              value={vaeScaleFactor}
              onChange={(e) => setVaeScaleFactor(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Overlap Ratio: {overlapRatio}</label>
            <input 
              type="range" 
              min="2" 
              max="8" 
              step="1"
              value={overlapRatio}
              onChange={(e) => setOverlapRatio(parseInt(e.target.value))}
              className="w-full" 
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Tile Order</label>
            <select 
              value={tileOrder}
              onChange={(e) => setTileOrder(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded">
              <option value="corner_to_center">Corner to Center</option>
              <option value="center_to_corner">Center to Corner</option>
            </select>
          </div>
          
          <div className="mt-4 bg-gray-100 p-3 rounded text-sm">
            <h4 className="font-semibold">Calculated Values:</h4>
            <p>Latent Size: {latentSize.toFixed(1)}</p>
            <p>Overlap Size (latent): {overlapSizeLatent.toFixed(1)}</p>
            <p>Latent Patch Size: {latentPatchSize.toFixed(1)}</p>
            <p>Total Tiles: {tiles.length}</p>
          </div>
          
          <div className="mt-4">
            <h4 className="font-semibold">Legend:</h4>
            <div className="flex gap-2 mt-1">
              <span className="inline-block w-4 h-4 bg-blue-500 bg-opacity-30 border border-black"></span>
              <span>Grid Tiles</span>
            </div>
            <div className="flex gap-2 mt-1">
              <span className="inline-block w-4 h-4 bg-green-500 bg-opacity-30 border border-black"></span>
              <span>Edge Tiles</span>
            </div>
            <div className="flex gap-2 mt-1">
              <span className="inline-block w-4 h-4 bg-red-500 bg-opacity-30 border border-black"></span>
              <span>Corner Tile</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TileOrderingVisualization;