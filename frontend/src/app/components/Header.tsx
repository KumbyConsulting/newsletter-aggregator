'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Layout, Menu, Button, Dropdown, Avatar, Space, Badge, Typography, Drawer } from 'antd';
import type { MenuProps } from 'antd';
import { 
  MenuOutlined, 
  HomeOutlined, 
  SearchOutlined, 
  BellOutlined,
  UserOutlined,
  FileTextOutlined,
  TagOutlined,
  InfoCircleOutlined,
  BulbOutlined,
  SyncOutlined
} from '@ant-design/icons';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import './Header.css';

const { Header: AntHeader } = Layout;
const { Text } = Typography;

type MenuItem = Required<MenuProps>['items'][number];

interface HeaderProps {
  onUpdateClick?: () => void;
}

export default function Header({ onUpdateClick }: HeaderProps = {}) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [current, setCurrent] = useState('');
  const [screenWidth, setScreenWidth] = useState(1200); // Default to desktop
  const pathname = usePathname();

  // Update current based on pathname
  useEffect(() => {
    if (pathname) {
      const path = pathname.split('/')[1] || 'home';
      setCurrent(path);
    }
  }, [pathname]);

  // Handle resize for responsive layout
  useEffect(() => {
    // Set initial width
    if (typeof window !== 'undefined') {
      setScreenWidth(window.innerWidth);
    }

    const handleResize = () => {
      setScreenWidth(window.innerWidth);
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  const isMobile = screenWidth < 768;

  // Define menu items 
  const menuItems = useMemo<MenuItem[]>(() => [
    {
      key: 'home',
      icon: <HomeOutlined />,
      label: <Link href="/">Home</Link>,
    },
    {
      key: 'topics',
      icon: <TagOutlined />,
      label: <Link href="/topics">Topics</Link>,
    },
    {
      key: 'sources',
      icon: <FileTextOutlined />,
      label: <Link href="/sources">Sources</Link>,
    },
    {
      key: 'rag',
      icon: <BulbOutlined />,
      label: <Link href="/rag">AI Insights</Link>,
    },
    {
      key: 'about',
      icon: <InfoCircleOutlined />,
      label: <Link href="/about">About</Link>,
    },
  ], []);

  // Mobile menu items with the update option
  const mobileMenuItems = useMemo<MenuItem[]>(() => [
    ...menuItems,
    ...(onUpdateClick ? [{
      key: 'update',
      icon: <SyncOutlined />,
      label: <a onClick={() => { 
        setIsMobileMenuOpen(false);
        onUpdateClick();
      }}>Update Articles</a>,
    }] : []),
  ], [menuItems, onUpdateClick]);

  const userMenuItems = useMemo<MenuProps['items']>(() => [
    {
      key: 'profile',
      label: 'Profile',
    },
    {
      key: 'settings',
      label: 'Settings',
    },
    {
      key: 'admin',
      label: 'Admin Panel',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      label: 'Logout',
    },
  ], []);

  // Memoize processed mobile menu items
  const processedMobileMenuItems = useMemo(() => {
    return mobileMenuItems.map(item => ({
      ...item,
      style: {
        borderRadius: '0.5rem',
        margin: '4px 0',
        transition: 'var(--transition-base, all 0.2s ease)',
      }
    }));
  }, [mobileMenuItems]);

  return (
    <AntHeader className="kumby-header" role="banner" aria-label="Site header">
      <div className="header-container">
        {/* Logo and Navigation */}
        <div className="header-left">
          <div className="logo-wrapper">
            <Link href="/" className="logo-link" aria-label="Kumby Consulting - Home">
              <div className="kumby-logo" aria-hidden="true">
                K
              </div>
              {!isMobile && (
                <Text className="brand-text">
                  Kumby Consulting
                </Text>
              )}
            </Link>
          </div>
          
          {/* Desktop Navigation */}
          {!isMobile && (
            <Menu
              mode="horizontal"
              selectedKeys={[current]}
              items={menuItems}
              theme="dark"
              className="desktop-menu"
              role="navigation"
              aria-label="Main navigation"
            />
          )}
        </div>
        
        {/* Right Side Actions */}
        <div className="header-right">
          <Space size={16}>
            {/* Update Button */}
            {!isMobile && onUpdateClick && (
              <Button 
                type="primary" 
                icon={<SyncOutlined />} 
                onClick={onUpdateClick}
                className="btn-custom"
                aria-label="Update articles"
              >
                Update
              </Button>
            )}
            
            {/* Desktop Actions */}
            {!isMobile && (
              <>
                <Button 
                  type="text"
                  icon={<SearchOutlined />}
                  className="action-btn"
                  aria-label="Search"
                />
                
                <Badge count={3} size="small" color="var(--primary-color, #00405e)">
                  <Button 
                    type="text" 
                    icon={<BellOutlined />}
                    className="action-btn"
                    aria-label="Notifications"
                  />
                </Badge>
                
                <Dropdown
                  menu={{ items: userMenuItems }}
                  trigger={['click']}
                  placement="bottomRight"
                >
                  <Button
                    type="text"
                    className="action-btn"
                    aria-label="User menu"
                    icon={
                      <Avatar 
                        size="small" 
                        icon={<UserOutlined />} 
                        style={{ backgroundColor: 'var(--primary-color, #00405e)' }} 
                      />
                    }
                  />
                </Dropdown>
              </>
            )}
            
            {/* Mobile Menu Toggle */}
            {isMobile && (
              <Button 
                type="text" 
                icon={<MenuOutlined />} 
                onClick={() => setIsMobileMenuOpen(true)}
                className="action-btn"
                aria-label="Open menu"
                aria-expanded={isMobileMenuOpen}
              />
            )}
          </Space>
        </div>
      </div>
      
      {/* Mobile Drawer Menu */}
      <Drawer
        title={
          <div className="drawer-header">
            <div className="kumby-logo" aria-hidden="true">K</div>
            <Text className="brand-text">
              Kumby Consulting
            </Text>
          </div>
        }
        placement="right"
        closable={true}
        onClose={() => setIsMobileMenuOpen(false)}
        open={isMobileMenuOpen}
        width={280}
        styles={{
          header: { 
            borderBottom: '1px solid var(--border-color, rgba(0, 64, 94, 0.1))',
            padding: '16px'
          },
          body: { padding: 0 },
          mask: { 
            background: 'rgba(0, 0, 0, 0.5)', 
            backdropFilter: 'blur(4px)' 
          }
        }}
        className="kumby-drawer"
        aria-label="Mobile navigation menu"
      >
        <Menu 
          mode="vertical" 
          selectedKeys={[current]}
          items={processedMobileMenuItems}
          theme="light"
          className="mobile-menu"
          role="navigation"
        />
        <div className="mobile-menu-actions">
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button 
              icon={<SearchOutlined style={{ marginRight: '8px', fontSize: '16px' }} />} 
              block
              className="btn-secondary"
              aria-label="Search"
            >
              Search
            </Button>
            <Button 
              icon={<BellOutlined style={{ marginRight: '8px', fontSize: '16px' }} />} 
              block
              className="btn-secondary"
              aria-label="Notifications"
            >
              Notifications
              <Badge count={5} style={{ marginLeft: 8 }} color="var(--primary-color, #00405e)" />
            </Button>
            <Button 
              icon={<UserOutlined style={{ marginRight: '8px', fontSize: '16px' }} />}
              block
              className="btn-secondary"
              aria-label="Account settings"
            >
              Account
            </Button>
          </Space>
        </div>
      </Drawer>
    </AntHeader>
  );
} 