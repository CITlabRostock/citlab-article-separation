from citlab_python_util.geometry.polygon import string_to_poly
from citlab_python_util.io.file_loader import load_text_file
from citlab_python_util.parser.xml.page.page import Page


def get_article_polys_from_file(poly_file_name):
    """ Load polygons from a txt file or a PageXml file `poly_file_name`, split them up into articles
    (marked by empty lines in txt file (one polygon per line), marked by custom tag in PageXml file)
    and save them as `Polygon` objects.

    NOTE: In the case of txt files, we dont distinguish between the "None" and the regular article classes!!!

    :param poly_file_name: path to the txt or PageXml file holding the polygons
    :type poly_file_name: str
    :return: a triple containing two lists of lists of polygons, where each sublist corresponds to an article and a
             boolean value representing if the polygons loaded without errors
    """
    if poly_file_name.endswith(".txt"):
        try:
            poly_strings = load_text_file(poly_file_name)
        except Exception as e:
            # Cannot load txt file or cannot extract the articles from the txt
            print(e)
            return None, None, True

        if len(poly_strings) == 0:
            # Cannot load txt file or cannot extract the articles from the txt
            print("No article baselines found.")
            return None, None, True

        poly_strings.append("\n")
        res_with_none = []
        article_polys = []

        for poly_string in poly_strings:
            if poly_string == "\n":
                res_with_none.append(article_polys)
                article_polys = []
            else:
                poly = string_to_poly(str(poly_string))
                article_polys.append(poly)

        # no difference between "None" and regular article classes
        res_without_none = res_with_none
        return res_without_none, res_with_none, False

    if poly_file_name.endswith(".xml"):
        try:
            page = Page(poly_file_name)
            ad = page.get_article_dict()
        except Exception as e:
            # Cannot load PageXml file or cannot extract the articles from the PageXml
            print(e)
            return None, None, True

        # with "None" class
        res_with_none = []
        for a_polys in ad.values():
            a = []
            for a_poly in a_polys:
                if a_poly.baseline:
                    try:
                        a.append(a_poly.baseline.to_polygon())
                    except(AttributeError):
                        print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(
                            a_poly.id))
                        continue

                    # a.append(a_poly.baseline.to_polygon())
                else:
                    print(f"No baseline found: Skipping text line with id '{a_poly.id}' from file '{poly_file_name}'.")
            res_with_none.append(a)

        # without "None" class
        res_without_none = []
        for article_id in ad:
            if article_id is None:
                continue

            for a_poly in ad[article_id]:
                try:
                    res_without_none.append(a_poly.baseline.to_polygon())
                except(AttributeError):
                    print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(
                        a_poly.id))
                    continue

            # res_without_none.append([a_poly.baseline.to_polygon() for a_poly in ad[article_id]])

        if len(res_without_none) == 0:
            # Cannot load PageXml file or cannot extract the articles from the PageXml
            print("No article baselines (except such in \"None\" class) found.")
            return None, res_with_none, True

        return res_without_none, res_with_none, False