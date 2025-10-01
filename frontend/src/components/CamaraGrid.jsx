import React from "react";

function CameraView({ cameras }) {
  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {cameras.map((cam) => (
        <div key={cam.id} className="bg-black rounded shadow-lg p-2">
          <h3 className="text-white text-center">{cam.name}</h3>
          <img
            src={cam.url}
            alt={cam.name}
            className="w-full h-64 object-cover"
          />
          <p className="text-sm text-gray-300 mt-2">
            {cam.location} – {cam.coordinates}
          </p>
        </div>
      ))}
    </div>
  );
}

export default CameraView;
