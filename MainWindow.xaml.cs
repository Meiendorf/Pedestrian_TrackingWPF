using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Media;
using Emgu.CV;
using Emgu.CV.Tracking;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Cvb;
using System.IO;
using System.Drawing;
using System.Threading;
using System.Collections;
using Emgu.CV.Util;
using DSize = System.Drawing.Size;
using System.Text.RegularExpressions;
using System.Diagnostics;

namespace Pedestrians_Tracking
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    
    public partial class MainWindow
    {
        #region Variables
        //Init values
        //Haar cascade settings
        private static float HScaleFactor = 1.3f;
        private static int HMinNeighbors = 5;
        private static DSize HMinSize = new DSize(60, 60);
        private static DSize HMaxSize = new DSize(190, 190);
        //Hog detector settings
        private static float HogScaleFactor = 1.3f;
        private static DSize HogWinStride = new DSize(3, 3);
        //Во сколько раз необходимо увеличить изображение
        private static double DisplayImageScale = 2.2;
        //Максимальное число циклов, после которого необходимо сделать детекцию еще раз
        private static int VerNum = 25;
        //Текущий тип детектора
        private static InitTypes InType = InitTypes.Haar;
        //Файл, из которого будут браться кадры
        private static string FileName = Path.GetFullPath("../../ped2.avi");
        //Флаг, отображающий работу процесса трекинга
        private static bool TrackingActive = false;
        //Переменная, считающая кол-во циклов до переинициализации трекера
        private static int NeedReinit = 0;
        //Флаг, отображающий, была ли реинициализация не предыдущем цикле
        private static bool IsNew = true;
        //Список, хранящий в себе списки выделений объектов, за которыми ведется слежка
        private static List<List<Rectangle>> AllBoxes = new List<List<Rectangle>>();
        //Main objects
        //Объект для потокобезопасности основного потока
        private static object locker = new object();
        //Свойство для вывода изображения в окно
        public static Mat MainWindowImage
        {
            get
            {
                //Конвертация из BitmapImage в Mat
                var bmp = ConvertToBitmap(((App.Current.MainWindow as MainWindow).MainScreen.Source as BitmapImage));
                return new Image<Bgr, Byte>(bmp).Mat;
            }
            set
            {
                //Конвертация из Mat в BitmapImage
                App.Current.Dispatcher.BeginInvoke(new Action(delegate () 
                {
                    BitmapImage img = ConvertToBitmapImage(value.ToImage<Bgr, Byte>().Bitmap);
                    (App.Current.MainWindow as MainWindow).MainScreen.Source = img;
                }));
                
            }
        }
        //Объект для захвата кадров из видео
        public static VideoCapture Capture { get; set; } = new VideoCapture();
        //Класс для работы с KFC трекерами
        public static MultiTracker Tracker { get; set; } = new MultiTracker();
        //Каскад Хаара для выделения области интереса
        //Более быстрый метод, но менее точный
        public static CascadeClassifier Haar { get; set; }
        //Хог дескриптор с SVM моделью для выделения области интереса
        //Более ресурсопотребительный, но зато более точный
        public static HOGDescriptor Hog { get; set; } = new HOGDescriptor();
        //Поток, в котором будет выполняться трекинг
        public static Thread MainLoopThread { get; set; }
        #endregion
        #region Imaging
        //Метод для реинициализации трекера, принимает в себя тип детектирования и изображение
        //Возвращает инициализированный трекер
        public static MultiTracker Reinit_Tracker(InitTypes type, Mat frame)
        {
            //Результаты детектирования для иницилазиции KFC трекеров
            List<Rectangle> Result = new List<Rectangle>();
            //Детектирование с помощью каскадов Хаара
            if (type == InitTypes.Haar)
            {
                var Rects = Haar.DetectMultiScale(frame, HScaleFactor, HMinNeighbors, HMinSize, HMaxSize);
                foreach(var rect in Rects)
                {
                    if (rect != null)
                    {
                        Result.Add(rect);
                    }
                }
            }
            //Детектирования с помощью Хог дескриптора
            else
            {
                var RowRects = Hog.DetectMultiScale(frame, 0, HogWinStride, scale: HogScaleFactor);
                foreach(var rowRect in RowRects)
                {
                    Result.Add(rowRect.Rect);
                }
            }
            //Инициализация KCF трекерами
            var TTracker = new MultiTracker();
            foreach (var rect in Result)
            {
                TTracker.Add(new TrackerKCF(), frame, rect);
            }
            return TTracker;
        }
        //Получение изображения с объекта видеозахвата
        public static Mat GetCameraPicture()
        {
            //Переменная для хранения исходного изображения
            var RawFrame = new Mat();
            //Захват изображения
            Capture.Read(RawFrame);
            //Масштабирование
            var RawImage = RawFrame.ToImage<Bgr, Byte>();
            RawImage = RawImage.Resize(DisplayImageScale, Inter.Linear);
            //Возврат
            return RawImage.Mat;
        }
        //Конвертация Bitmap в BitmapImage
        public static BitmapImage ConvertToBitmapImage(Bitmap bmp)
        {
            using (var stream = new MemoryStream())
            {
                bmp.Save(stream, System.Drawing.Imaging.ImageFormat.Jpeg);
                BitmapImage im = new BitmapImage();
                im.BeginInit();
                im.StreamSource = new MemoryStream(stream.ToArray());
                im.EndInit();
                return im;
            }
        }
        //Конвертация BitmapImage в Bitmap
        public static Bitmap ConvertToBitmap(BitmapImage bitmapImage)
        {
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapImage));
                enc.Save(outStream);
                System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap(outStream);
                return new Bitmap(bitmap);
            }
        }
        //Получение настроек из файла
        public static void GetSettingsFromFile()
        {
            HScaleFactor = Properties.Settings.Default.HScaleFactor;
            int min = Properties.Settings.Default.HMinSize;
            int max = Properties.Settings.Default.HMaxSize;
            HMinSize = new DSize(min, min);
            HMaxSize = new DSize(max, max);
            HMinNeighbors = Properties.Settings.Default.HMinNeighbors;
            HogScaleFactor = Properties.Settings.Default.HogScaleFactor;
            int win = Properties.Settings.Default.HogWinStride;
            HogWinStride = new DSize(win, win);
        }
        #endregion
        //Основной цикл трекинга
        public static void MainLoop()
        {
            //Обеспечение потокобезопасности
            lock (locker)
            {
                //Продолжение работы, до нажатия кнопки Stop или же до окончания кадров в файле
                while (TrackingActive || Capture.IsOpened)
                {
                    //Заддержка для плавности работы
                    Thread.Sleep(10);
                    //Номер выделяемого объекта
                    int nBox = 0;
                    //Получение изображения с камеры
                    Mat frame = GetCameraPicture();
                    //Переменная для хранения данных, получаемых в результате работы трекеров
                    VectorOfRect RawRects = new VectorOfRect();
                    //Получение данных для нового изображения
                    bool state = Tracker.Update(frame, RawRects);
                    //Преобразование в массив точек
                    Rectangle[] Rects = RawRects.ToArray();
                    //Проверка на надобность реинициализации
                    if ((!state) || (Rects.Length == 0) || (NeedReinit > VerNum))
                    {
                        //Обнуление счетчика 
                        NeedReinit = 0;
                        //Вывод изображения
                        MainWindowImage = frame;
                        //Реинициализация
                        Tracker = Reinit_Tracker(InType, frame);
                        //Удаление всех данных о прошлых объектах
                        AllBoxes.Clear();
                        //Установка флага
                        IsNew = true;
                        //Переход к следующей итерации
                        continue;
                    }
                    //Если же реинициализация не нужна, переходим к отрисовке выделений и поиску скорости
                    foreach (var rect in Rects)
                    {
                        //Скорость выделяемого объекта
                        float speed = 0;
                        //Проверка на то, была ли реинициализация на прошлой итерации
                        //Если да, добавление нового списка в главный
                        if (IsNew)
                        {
                            AllBoxes.Add(new List<Rectangle>());
                        }
                        //Проверка на то, есть ли в текущем списке прямоугольников значения
                        //Если да, то можно найти условную скорость объекта
                        if (AllBoxes[nBox].Count != 0)
                        {
                            //Получение точек прошлой итерации
                            Rectangle oldRect = AllBoxes[nBox][AllBoxes[nBox].Count - 1];
                            //Нахождение общего значения, на основе которого можно будет вычислить отклонение
                            float oldSpeed = (oldRect.X + oldRect.Y + oldRect.Width + oldRect.Height) / 4;
                            speed = Convert.ToSingle((rect.X + rect.Y + rect.Width + rect.Height)) / 4;
                            //Вычисление отклонения от предыдущего изображения. Чем оно больше - тем выше была скорость объекта
                            speed = Math.Abs(oldSpeed - speed);
                        }
                        //Добавление текущих точек в список
                        AllBoxes[nBox].Add(rect);
                        //Точка, где будет отрисована скорость
                        var TextLocation = new System.Drawing.Point(rect.X - 20, rect.Y - 30);
                        //Отрисовка скорости
                        CvInvoke.PutText(frame, Convert.ToString(speed), TextLocation, FontFace.HersheyScriptSimplex, 1, new MCvScalar(0, 0, 0), 3);
                        //Отрисовка найденного прямоугольника
                        CvInvoke.Rectangle(frame, rect, new MCvScalar(0, 255, 0), 2);
                        //Отрисовка траектории на основе прошлых итерации
                        foreach (var pBox in AllBoxes[nBox])
                        {
                            var PointLocation = new System.Drawing.Point(pBox.X + Convert.ToInt32(pBox.Width / 2), pBox.Y + pBox.Height);
                            CvInvoke.Circle(frame, PointLocation, 3, new MCvScalar(0, 0, 255));
                        }
                        //Переход к следующему объекту
                        nBox++;
                    }
                    //Смена флага
                    IsNew = false;
                    //Увеличение счетчика
                    NeedReinit++;
                    //Вывод отрисованного изображения на экран
                    MainWindowImage = frame;
                }
            }
        }
        #region UI
        //Конструктор главного окна
        public MainWindow()
        {
            InitializeComponent();
        }
        //Обработчик на событие загрузки окна
        private void MetroWindow_Loaded(object sender, RoutedEventArgs e)
        {
            //Initialization
            GetSettingsFromFile();
            //Первоначальная инициализация объекта захвата изображений для тесов
            Capture = new VideoCapture(FileName);
            //Инициализация каскада Хаара
            Haar = new CascadeClassifier(Path.GetFullPath(Properties.Settings.Default.HaarPath));
            //Инициализация SVM детектора
            Hog.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
            //Перерисовка некоторых значений на экране
            RedrawComponents();
        }
        //Обработчик на кнопку старта
        private void StartBut_Click(object sender, RoutedEventArgs e)
        {
            //Проверки на возможности запуска
            //Если файл не выбран или поврежден, запуститься нельзя
            if(!Capture.IsOpened || FileName == "")
            {
                MessageBox.Show("File open error.", "Error");
                return;
            }
            //Если главный цикл уже запущен, запускать его заново нет смысла
            if(TrackingActive)
            {
                return;
            }
            //Инициализация потока трекинга
            MainLoopThread =  new Thread(new ThreadStart(MainLoop));
            //Делаем его фоновым, что-бы можно было спокойно закрывать окно
            MainLoopThread.IsBackground = true;
            //Показываем, что цикл запущен
            TrackingActive = true;
            //Запускаем
            MainLoopThread.Start();
        }
        //Обработчик на кнопку открытия файла
        private void OpenBut_Click(object sender, RoutedEventArgs e)
        {
            //Открытия файла через стандартный проводник системы
            var FileDialog = new System.Windows.Forms.OpenFileDialog();
            if (FileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                //Присваивание имени файла переменной
                FileName = FileDialog.FileName;
                //Открытия файла в объекте захвата
                Capture = new VideoCapture(FileName);
            }

        }
        //Обработчик на кнопку завершения цикла
        private void StopBut_Click(object sender, RoutedEventArgs e)
        {
            //Меняем флаг, в результате чего фоновый процесс трекинга завершится
            TrackingActive = false;
            //Реинициализация объекта захвата
            Capture = new VideoCapture(FileName);
        }
        //Обработчик на кнопку сохранения настроек
        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            //Проверка введеных данных на валидность
            try
            {
                VerNum = Convert.ToInt32(VerNumBox.Text);
                DisplayImageScale = Convert.ToDouble(ScaleBox.Text);
            }
            catch
            {
                MessageBox.Show("Wrong charactres. Try again.", "Error");
            }
            //Перерисовка интерфейса в соответствии с переменными
            RedrawComponents();
        }
        //Обработчик на смену значения свитча метода детектирования
        private void InitModeSwitch_Click(object sender, RoutedEventArgs e)
        {
            //Если он активен, то присваиваем текущему методу каскад
            if((bool)InitModeSwitch.IsChecked)
            {
                InType = InitTypes.Haar;
            }
            //Если же нет, то Хог дескриптор
            else
            {
                InType = InitTypes.Hog;
            }
            //Перерисовываем
            RedrawComponents();
        }
        //Метод для перерисовки окна, основываясь на значениях переменных
        public void RedrawComponents()
        {
            //Перерисовка свитча
            if (InType == InitTypes.Haar)
            {
                InitModeSwitch.IsChecked = true;
            }
            else
            {
                InitModeSwitch.IsChecked = false;
            }
            //Перерисовка текстовых полей
            ScaleBox.Text = DisplayImageScale.ToString();
            VerNumBox.Text = VerNum.ToString();
        }
        #endregion
    }
    //Перечисление типов детектирования
    public enum InitTypes
    {
        //Хог дескриптор
        Hog,
        //Каскады Хаара
        Haar
    }
}
