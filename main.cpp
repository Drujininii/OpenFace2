#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>

using namespace dlib;
using namespace std;

#include <string>
#include <sstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>


#define INPUT_IMAGE "./image3.jpg"

template <typename T>
std::string toString(T val)
{
    std::ostringstream oss;
    oss<< val;
    return oss.str();
}

template<typename T>
T fromString(const std::string& s)
{
    std::istringstream iss(s);
    T res;
    iss >> res;
    return res;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.


        //вот это выпили, как только все запустится
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        //каждое лицо в этой проге берут в прямоугольник и находят на нем 68 точек
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        image_window win, win_faces;//это можно к чертям удалить
        //цикл по всем фоткам, которые ты дал в аргументе
        for (int i = 2; i < argc; ++i)
        {
            const std::string image_name = INPUT_IMAGE;
            cv::Mat image;
            image = cv::imread(image_name);
            dlib::cv_image<bgr_pixel> cvImage(image);

            //опеределяем прямоугольники для каждого лица
            std::vector<rectangle> dets = detector(cvImage);
            cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)//для всех обнаруженных лиц
            {
                full_object_detection shape = sp(cvImage, dets[j]);//здесь получаем объект из 68 точек на лице
                shapes.push_back(shape);
            }


            //здесь рисуется
            win.clear_overlay();
//первые 16 точек - это овал лица
            win.set_image(cvImage);
            //пронумеруем точки
            size_t min_num_point = 0;

            for (size_t k = min_num_point; k < shapes[0].num_parts(); k++) {//в цикле вывел тебе пронумерованные точки.
                //а то так и не нашел инфу на них
                std::string number_of_point = toString(k);
                win.add_overlay(
                        dlib::image_window::overlay_rect(shapes[0].part(k), rgb_pixel(255, 0, 0), number_of_point));
//            win.add_overlay(render_face_detections(shapes)); //этой строкой можно обрисовать маску на все лицо
            }
            cout << "Hit enter to process the next image..." << endl;
            cin.get();

        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}



//смотри, отсюда можно выкинуть рисовалку и оставить только получение точек нужных.если у тебя все соберется,
//для удобства нарисовал цифры и отметил точки.
//координаты точки ты получаешь по строке:
//point = shapes[0].part(i) индекс у shapes - это номер лица(0, т.к. вроде оно одно у нас), а у part - номер точки
//вроде 37 и 43 похожи на центр глаза.
//думаю, нужно еще point из dlib переделать в поинт opencv. хотя там есть адекватная совместимость между ними
