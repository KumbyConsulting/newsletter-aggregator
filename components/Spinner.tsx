import React from 'react';

export const Spinner: React.FC = () => (
    <div className="spinner" role="status">
        <div className="spinner-inner"></div>
        <span className="sr-only">Loading...</span>
    </div>
); 