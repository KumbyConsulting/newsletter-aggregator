import React from 'react';
import { App as AntdApp } from 'antd';
import { Toaster } from 'react-hot-toast';

interface AppProps {
    children: React.ReactNode;
}

export const App: React.FC<AppProps> = ({ children }) => {
    return (
        <AntdApp>
            {children}
            <Toaster position="top-right" />
        </AntdApp>
    );
}; 