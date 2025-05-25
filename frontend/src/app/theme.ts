import { ThemeConfig } from 'antd';

export const theme: ThemeConfig = {
  token: {
    colorPrimary: '#00405e',
    colorPrimaryHover: '#fae061',
    colorPrimaryActive: '#003c56',
    borderRadius: 12,
    fontFamily: 'DM Sans, Arial, Helvetica, sans-serif',
  },
  components: {
    Layout: {
      bodyBg: '#f1f2e7',
      headerBg: '#fff',
      footerBg: '#f5f5f5',
      headerPadding: '0 50px',
      headerHeight: 64,
    },
    Card: {
      colorBorderSecondary: 'rgba(0, 64, 94, 0.1)',
      borderRadiusLG: 20,
      fontFamily: 'DM Sans, Arial, Helvetica, sans-serif',
    },
    Button: {
      colorPrimary: '#00405e',
      colorPrimaryHover: '#fae061',
      colorPrimaryActive: '#003c56',
      borderRadius: 12,
      fontFamily: 'DM Sans, Arial, Helvetica, sans-serif',
    },
    Table: {
      fontSize: 14,
      fontFamily: 'DM Sans, Arial, Helvetica, sans-serif',
    },
    Pagination: {
      itemActiveBg: '#00405e',
      fontFamily: 'DM Sans, Arial, Helvetica, sans-serif',
    },
  },
}; 