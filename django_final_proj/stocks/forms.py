from django import forms

class StockForm(forms.Form):
    input_string = forms.CharField(max_length=100)
    news_input = forms.CharField(max_length=10000, widget=forms.widgets.Textarea())

    # def __init__(self, *args, **kwargs):
    #     super(StockForm, self).__init__(*args, **kwargs) # Call to ModelForm constructor
    #     self.fields['news_input'].widget.attrs['style']  = 'width:800px; height:80px;'
