import { ThemeConfig } from 'antd';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const theme: ThemeConfig = {
  token: {
    colorPrimary: '#00405e',
    borderRadius: 6,
    fontFamily: inter.style.fontFamily,
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
    },
    Button: {
      colorPrimaryHover: '#00567e',
      colorPrimaryActive: '#003c56',
    },
    Table: {
      fontSize: 14,
    },
    Pagination: {
      itemActiveBg: '#00405e',
    },
  },
}; 