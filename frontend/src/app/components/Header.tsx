'use client';

import React from 'react';
import { useState } from 'react';
import Link from 'next/link';
import { 
  Layout, 
  Menu, 
  Button, 
  Space, 
  Typography, 
  Drawer, 
  Divider, 
  Tooltip
} from 'antd';
import type { MenuProps } from 'antd';
import { 
  MenuOutlined, 
  CloseOutlined, 
  SyncOutlined, 
  SearchOutlined, 
  BellOutlined, 
  UserOutlined,
  AppstoreOutlined,
  GlobalOutlined,
  ReadOutlined,
  InfoCircleOutlined,
  HomeOutlined
} from '@ant-design/icons';
import ThemeToggle from './ui/ThemeToggle';

const { Header: AntHeader } = Layout;
const { Text } = Typography;

interface HeaderProps {
  onUpdateClick?: () => void;
}

type MenuItem = Required<MenuProps>['items'][number];

export default function Header({ onUpdateClick }: HeaderProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Define menu items for desktop and mobile
  const menuItems: MenuItem[] = [
    {
      key: 'dashboard',
      icon: <HomeOutlined />,
      label: <Link href="/">Dashboard</Link>,
    },
    {
      key: 'topics',
      icon: <AppstoreOutlined />,
      label: <Link href="/topics">Topics</Link>,
    },
    {
      key: 'sources',
      icon: <GlobalOutlined />,
      label: <Link href="/sources">Sources</Link>,
    },
    {
      key: 'knowledge',
      icon: <ReadOutlined />,
      label: <Link href="/knowledge-base">Knowledge Base</Link>,
    },
    {
      key: 'about',
      icon: <InfoCircleOutlined />,
      label: <Link href="/about">About</Link>,
    }
  ];

  // Mobile menu items with the update option
  const mobileMenuItems: MenuItem[] = [
    {
      key: 'dashboard',
      icon: <HomeOutlined />,
      label: <Link href="/" onClick={() => setIsMobileMenuOpen(false)}>Dashboard</Link>,
    },
    {
      key: 'topics',
      icon: <AppstoreOutlined />,
      label: <Link href="/topics" onClick={() => setIsMobileMenuOpen(false)}>Topics</Link>,
    },
    {
      key: 'sources',
      icon: <GlobalOutlined />,
      label: <Link href="/sources" onClick={() => setIsMobileMenuOpen(false)}>Sources</Link>,
    },
    {
      key: 'knowledge',
      icon: <ReadOutlined />,
      label: <Link href="/knowledge-base" onClick={() => setIsMobileMenuOpen(false)}>Knowledge Base</Link>,
    },
    {
      key: 'about',
      icon: <InfoCircleOutlined />,
      label: <Link href="/about" onClick={() => setIsMobileMenuOpen(false)}>About</Link>,
    },
    {
      type: 'divider',
    } as MenuItem,
    {
      key: 'update',
      icon: <SyncOutlined />,
      label: 'Update Feeds',
      onClick: () => {
        if (onUpdateClick) onUpdateClick();
        setIsMobileMenuOpen(false);
      },
    },
  ];

  return (
    <AntHeader style={{ 
      position: 'sticky', 
      top: 0, 
      zIndex: 50, 
      padding: 0, 
      background: 'rgba(255, 255, 255, 0.8)',
      backdropFilter: 'blur(5px)',
      borderBottom: '1px solid #f0f0f0'
    }} className="dark:bg-gray-900/80 dark:border-gray-800">
      <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '0 16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '64px' }}>
          {/* Logo and main navigation */}
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ marginRight: '24px' }}>
              <Link href="/" style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ 
                  fontSize: '24px', 
                  fontWeight: 'bold', 
                  background: 'linear-gradient(to right, var(--primary-color), var(--primary-light))', 
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  marginRight: '4px'
                }}>
                  Kumby
                </span>
                <span style={{ 
                  fontSize: '24px', 
                  fontWeight: 300, 
                  color: 'rgba(0, 0, 0, 0.65)'
                }} className="dark:text-gray-300">
                  Newsboard
                </span>
              </Link>
            </div>
            
            {/* Desktop Navigation */}
            <div className="hidden md:block">
              <Menu 
                mode="horizontal" 
                style={{ 
                  background: 'transparent', 
                  borderBottom: 'none'
                }}
                defaultSelectedKeys={['dashboard']}
                items={menuItems}
              />
            </div>
          </div>
          
          {/* Action Buttons - Desktop */}
          <div className="hidden md:flex md:items-center">
            <Space size="middle">
              <Tooltip title="Search">
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<SearchOutlined />} 
                  style={{ fontSize: '16px' }}
                />
              </Tooltip>
              
              <Tooltip title="Notifications">
                <Button 
                  type="text" 
                  shape="circle" 
                  icon={<BellOutlined />} 
                  style={{ fontSize: '16px' }}
                />
              </Tooltip>
              
              <ThemeToggle />
              
              <Button 
                type="primary"
                icon={<SyncOutlined />}
                onClick={onUpdateClick}
              >
                Update Feeds
              </Button>
              
              <Tooltip title="User Profile">
                <Button 
                  type="primary" 
                  shape="circle" 
                  icon={<UserOutlined />}
                  style={{ background: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}
                />
              </Tooltip>
            </Space>
          </div>
          
          {/* Mobile Menu Toggle */}
          <div className="md:hidden flex items-center">
            <ThemeToggle />
            
            <Button
              type="text"
              icon={isMobileMenuOpen ? <CloseOutlined /> : <MenuOutlined />}
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              style={{ marginLeft: '8px' }}
            />
          </div>
        </div>
      </div>

      {/* Mobile Menu Drawer */}
      <Drawer
        title="Menu"
        placement="right"
        onClose={() => setIsMobileMenuOpen(false)}
        open={isMobileMenuOpen}
        styles={{ body: { padding: 0 } }}
        width={280}
      >
        <Menu 
          mode="vertical" 
          style={{ border: 'none' }}
          defaultSelectedKeys={['dashboard']}
          items={mobileMenuItems}
        />
      </Drawer>
    </AntHeader>
  );
} 