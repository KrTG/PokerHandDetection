import cv2

class SlidingBuffer(list):
    """
    Klasa reprezentująca bufor przesuwny

    Metody
    ------
    @classmethod
    from_iterator(iterator)
        Tworzy nowy bufor z istniejącej sekwencji

    append(element)
        Dołącza nowy element do bufora, co powoduje usunięcie najstarszego elementu
    """
    
    @classmethod
    def from_iterator(cls, iterator):
        """
        Tworzy nowy bufor z istniejącej sekwencji

        Parametry
        ---------
        iterator: sekwencja, której elementami zostanie wypełniony bufor

        Wartość zwracana
        ----------------
        SlidingBuffer, nowy bufor
        """

        buffer = cls(0)
        for element in iterator:
            super(cls, buffer).append(element)
        if (len(buffer) == 0):
            raise ValueError("The iterator has to have at least one element")
        return buffer

    def __init__(self, size):
        """
        Parametry
        ---------
        size: int, wielkość bufora
        """
        super()
    
    def append(self, element):
        """
        Dołącza nowy element do bufora, po usunięciu najstarszego elementu
        i przesunięciu pozostałych w o jedno miejsce w lewo

        Parametry
        ---------
        element: dołączany element
        """

        for i in range(len(self) - 1):
            self[i] = self[i + 1]    

        self[len(self) - 1] = element

def show(image, scaling=1):
    to_show = cv2.resize(image, (0, 0), fx=scaling, fy=scaling)
    cv2.imshow("title", to_show)
    cv2.moveWindow("title", 0, 0)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()